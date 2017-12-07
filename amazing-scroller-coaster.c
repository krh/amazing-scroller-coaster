/*
 * Copyright (c) 2012 Arvin Schnell <arvin.schnell@gmail.com>
 * Copyright (c) 2012 Rob Clark <rob@ti.com>
 * Copyright (c) 2017 Google, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/* Based on kmscube. */

#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <poll.h>
#include <unistd.h>
#include <signal.h>
#include <termios.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/mman.h>
#include <linux/kd.h>
#include <linux/vt.h>
#include <linux/major.h>
#include <immintrin.h>
#include <math.h>

#include "uapi/i915_drm.h"

#include <stdbool.h>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <xf86drm.h>
#include <xf86drmMode.h>

#include <gbm.h>
#include <drm_fourcc.h>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#define container_of(ptr, type, member) ({				\
	void *__mptr = (void *)(ptr);					\
	((type *)(__mptr - offsetof(type, member))); })

struct scroller {
	int (*draw)(struct scroller *scroller, uint32_t offset_x, uint32_t offset_y);
};

struct egl {
	EGLDisplay display;
	EGLConfig config;
	EGLContext context;
	EGLSurface surface;

	PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT;
	PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR;
	PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR;
	PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES;
	PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR;
	PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR;
	PFNEGLWAITSYNCKHRPROC eglWaitSyncKHR;
	PFNEGLCLIENTWAITSYNCKHRPROC eglClientWaitSyncKHR;
	PFNEGLDUPNATIVEFENCEFDANDROIDPROC eglDupNativeFenceFDANDROID;
};

struct props {
	const char *name;
	uint32_t count;
	drmModePropertyRes **info;
};

struct drm {
	int fd;

	drmModeConnector *connector;
	struct props connector_props;

	drmModeCrtc *crtc;
	struct props crtc_props;

	drmModePlane *plane;
	struct props plane_props;

	int crtc_index;
	int kms_in_fence_fd;
	int kms_out_fence_fd;

	drmModeModeInfo *mode;
};

struct gpu_scroller {
	struct scroller base;

	struct gbm_device *dev;
	struct gbm_surface *surface;
	int width, height;
	struct gbm_bo *last_bo;

	struct drm drm;
	struct egl egl;

	GLuint program;
	/* uniform handles: */
	GLint texture;
	GLuint vbo;

	struct tile *tiles[64];
};

struct drm_fb {
	struct gbm_bo *bo;
	uint32_t fb_id;
};

struct scratch_bo {
	uint32_t gem_handle;
	uint32_t stride;
	uint32_t fb_id;
};

static int
safe_ioctl(int fd, unsigned long request, void *arg)
{
	int ret;

	do {
		ret = ioctl(fd, request, arg);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline uint32_t
align_u32(uint32_t u, uint32_t align)
{
	return (u + align - 1) & ~(align - 1);
}

static void
copy_linear_to_ymajor(void *dst, void *src, uint32_t stride, uint32_t height)
{
	int tile_stride = stride / 128;
	const int column_stride = 32 * 16;
	int columns = stride / 16;

	assert((stride & 127) == 0);

	for (int y = 0; y < height; y += 2) {
		int tile_y = y / 32;
		int iy = y & 31;
		void *s = src + y * stride;
		void *d = dst + tile_y * tile_stride * 4096 + iy * 16;

		for (int x = 0; x < columns; x++) {
			__m128i lo = _mm_load_si128((s + x * 16));
			__m128i hi = _mm_load_si128((s + x * 16 + stride));
			__m256i p = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
			_mm256_store_si256((d + x * column_stride), p);
		}
	}
}

static void
fill_tile(void *map, uint32_t width, uint32_t height, uint32_t stride)
{
	uint32_t bpp = 4;
	void *shadow;
	struct color { float r, g, b; } c[2] = {
		{ drand48(), drand48(), drand48() },
		{ drand48(), drand48(), drand48() },
	};

	const uint32_t point_count = rand() % 5 + 3;
	const float v1 = drand48() * 0.2 + 0.1;
	const float v0 = drand48() * 0.4;
	const float phase = drand48() * M_PI;

	shadow = malloc(height * stride * bpp);
	for (uint32_t y = 0; y < height; y++) {
		uint32_t *line = shadow + y * stride;
		for (uint32_t x = 0; x < width; x++) {
			float fx = (float) x - width / 2;
			float fy = (float) y - height / 2;
			float d = (fx * fx + fy * fy) / 300;
			float a = atan2f(fy, fx) + phase;
			float t = fmax(fmin(4 - d * (sin(a * point_count) * v1 + v0), 1.0f), 0.0f);

			uint32_t r = (c[0].r * t + c[1].r * (1 - t)) * 255 + 0.5f;
			uint32_t g = (c[0].g * t + c[1].g * (1 - t)) * 255 + 0.5f;
			uint32_t b = (c[0].b * t + c[1].b * (1 - t)) * 255 + 0.5f;

			line[x] = 0xff000000 | (b << 16) | (g << 8) | (r << 0);
		}
	}

	copy_linear_to_ymajor(map, shadow, stride, height);
	free(shadow);
}

struct tile {
	GLuint texture;
	uint32_t stride;
	int gem_handle;
};

static const uint32_t tile_width = 256, tile_height = 256;
static const uint32_t tile_row_stride = 29;

static struct tile *
create_tile(int fd, uint32_t width, uint32_t height)
{
	struct tile *tile;

	assert(width == align_u32(width, 32));
	assert(height == align_u32(height, 32));

	tile = malloc(sizeof(*tile));
	if (tile == NULL)
		return NULL;

	uint32_t bpp = 4;
	tile->stride = width * bpp; /* multiple of 128 */
	uint32_t size = tile->stride * height;

	struct drm_i915_gem_create_v2 gem_create = {
		.size = size
	};

	int ret = safe_ioctl(fd, DRM_IOCTL_I915_GEM_CREATE, &gem_create);
	if (ret < 0)
		return NULL;

	tile->gem_handle = gem_create.handle;

	/* Shouldn't have to set tiling, the modifier in import should
	 * override... but it doesn't. */
	do {
		struct drm_i915_gem_set_tiling set_tiling = {
			.handle = tile->gem_handle,
			.tiling_mode = I915_TILING_Y,
			.stride = tile->stride,
		};

		ret = safe_ioctl(fd, DRM_IOCTL_I915_GEM_SET_TILING, &set_tiling);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	struct drm_i915_gem_mmap gem_mmap = {
		.handle = tile->gem_handle,
		.offset = 0,
		.size = size,
		.flags = 0,
	};

	ret = safe_ioctl(fd, DRM_IOCTL_I915_GEM_MMAP, &gem_mmap);
	if (ret != 0)
		return NULL;

	void *map = (void *)(long)gem_mmap.addr_ptr;
	fill_tile(map, width, height, tile->stride);
	munmap(map, size);

	return tile;
}

static int
create_texture_for_tile(int fd, struct egl *egl, struct tile *tile,
			uint32_t width, uint32_t height)
{
	int ret;

	struct drm_prime_handle prime_handle = {
		.handle = tile->gem_handle,
		.flags = DRM_CLOEXEC,
	};

	ret = safe_ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime_handle);
	if (ret == -1)
		return -1;

	const EGLint attr[] = {
		EGL_WIDTH, width,
		EGL_HEIGHT, height,
		EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ABGR8888,
		EGL_DMA_BUF_PLANE0_FD_EXT, prime_handle.fd,
		EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
		EGL_DMA_BUF_PLANE0_PITCH_EXT, tile->stride,
		EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (uint32_t) I915_FORMAT_MOD_Y_TILED,
		EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (uint32_t) (I915_FORMAT_MOD_Y_TILED >> 32),
		EGL_NONE
	};

	glGenTextures(1, &tile->texture);

	EGLImage img = egl->eglCreateImageKHR(egl->display, EGL_NO_CONTEXT,
					      EGL_LINUX_DMA_BUF_EXT, NULL, attr);
	if (img == NULL) {
		printf("failed to create egl image\n");
		return -1;
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tile->texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	egl->glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img);

	egl->eglDestroyImageKHR(egl->display, img);

	return 0;
}

static void
drm_fb_destroy_callback(struct gbm_bo *bo, void *data)
{
	int drm_fd = gbm_device_get_fd(gbm_bo_get_device(bo));
	struct drm_fb *fb = data;

	if (fb->fb_id)
		drmModeRmFB(drm_fd, fb->fb_id);

	free(fb);
}

static struct drm_fb *
drm_fb_get_from_bo(struct gpu_scroller *scroller, struct gbm_bo *bo)
{
	int drm_fd = gbm_device_get_fd(gbm_bo_get_device(bo));
	struct drm_fb *fb = gbm_bo_get_user_data(bo);
	int ret = -1;

	if (fb)
		return fb;

	fb = calloc(1, sizeof *fb);
	fb->bo = bo;

	uint32_t width = gbm_bo_get_width(bo);
	uint32_t height = gbm_bo_get_height(bo);

	uint32_t handles[4] = { gbm_bo_get_handle(bo).u32, };
	uint32_t strides[4] = { gbm_bo_get_stride_for_plane(bo, 0), };
	uint32_t offsets[4] = { 0, };
	uint64_t modifiers[4] = { I915_FORMAT_MOD_Y_TILED, };

	ret = drmModeAddFB2WithModifiers(drm_fd, width, height,
			DRM_FORMAT_ABGR8888, handles, strides, offsets,
			modifiers, &fb->fb_id, DRM_MODE_FB_MODIFIERS);

	if (ret) {
		printf("failed to create fb: %s\n", strerror(errno));
		free(fb);
		return NULL;
	}

	gbm_bo_set_user_data(bo, fb, drm_fb_destroy_callback);

	return fb;
}

static uint32_t
find_crtc_for_connector(const struct drm *drm, const drmModeRes *resources,
			const drmModeConnector *connector) {
	int i;

	for (i = 0; i < connector->count_encoders; i++) {
		const uint32_t encoder_id = connector->encoders[i];
		drmModeEncoder *encoder = drmModeGetEncoder(drm->fd, encoder_id);

		if (encoder) {
			int i = __builtin_ffs(encoder->possible_crtcs) - 1;

			return resources->crtcs[i];
		}
	}

	/* no match found */
	return -1;
}

#define VOID2U64(x) ((uint64_t)(unsigned long)(x))

static int add_property(drmModeAtomicReq *req, struct props *props,
			uint32_t obj_id, const char *name, uint64_t value)
{
	unsigned int i;
	int prop_id = 0;

	for (i = 0 ; i < props->count ; i++) {
		if (strcmp(props->info[i]->name, name) == 0) {
			prop_id = props->info[i]->prop_id;
			break;
		}
	}

	if (prop_id < 0) {
		printf("no %s property: %s\n", props->name, name);
		return -EINVAL;
	}

	return drmModeAtomicAddProperty(req, obj_id, prop_id, value);
}

static int
drm_atomic_commit(struct drm *drm, uint32_t fb_id,
		  uint32_t x, uint32_t y,
		  uint32_t width, uint32_t height, uint32_t flags)
{
	drmModeAtomicReq *req;
	uint32_t plane_id = drm->plane->plane_id;
	uint32_t blob_id;
	uint32_t crtc_id = drm->crtc->crtc_id;
	int ret;

	req = drmModeAtomicAlloc();

	if (flags & DRM_MODE_ATOMIC_ALLOW_MODESET) {
		if (add_property(req, &drm->connector_props,
				 drm->connector->connector_id, "CRTC_ID",
				 crtc_id) < 0)
			return -1;

		if (drmModeCreatePropertyBlob(drm->fd, drm->mode, sizeof(*drm->mode),
					      &blob_id) != 0)
			return -1;

		if (add_property(req, &drm->crtc_props,
				 crtc_id, "MODE_ID", blob_id) < 0)
			return -1;

		if (add_property(req, &drm->crtc_props,
				 crtc_id, "ACTIVE", 1) < 0)
			return -1;
	}

	add_property(req, &drm->plane_props, plane_id, "FB_ID", fb_id);
	add_property(req, &drm->plane_props, plane_id, "CRTC_ID", crtc_id);
	add_property(req, &drm->plane_props, plane_id, "SRC_X", x << 16);
	add_property(req, &drm->plane_props, plane_id, "SRC_Y", y << 16);
	add_property(req, &drm->plane_props, plane_id, "SRC_W", width << 16);
	add_property(req, &drm->plane_props, plane_id, "SRC_H", height << 16);
	add_property(req, &drm->plane_props, plane_id, "CRTC_X", 100);
	add_property(req, &drm->plane_props, plane_id, "CRTC_Y", 100);
	add_property(req, &drm->plane_props, plane_id, "CRTC_W", width);
	add_property(req, &drm->plane_props, plane_id, "CRTC_H", height);

	if (drm->kms_in_fence_fd != -1) {
		add_property(req, &drm->crtc_props, crtc_id,
			     "OUT_FENCE_PTR", VOID2U64(&drm->kms_out_fence_fd));
		add_property(req, &drm->plane_props, plane_id,
			     "IN_FENCE_FD", drm->kms_in_fence_fd);
	}

	ret = drmModeAtomicCommit(drm->fd, req, flags, NULL);
	if (ret)
		goto out;

	if (drm->kms_in_fence_fd != -1) {
		close(drm->kms_in_fence_fd);
		drm->kms_in_fence_fd = -1;
	}

out:
	drmModeAtomicFree(req);

	return ret;
}

static EGLSyncKHR create_fence(const struct egl *egl, int fd)
{
	EGLint attrib_list[] = {
		EGL_SYNC_NATIVE_FENCE_FD_ANDROID, fd,
		EGL_NONE,
	};
	EGLSyncKHR fence = egl->eglCreateSyncKHR(egl->display,
			EGL_SYNC_NATIVE_FENCE_ANDROID, attrib_list);
	assert(fence);
	return fence;
}

/* Pick a plane.. something that at a minimum can be connected to
 * the chosen crtc, but prefer primary plane.
 *
 * Seems like there is some room for a drmModeObjectGetNamedProperty()
 * type helper in libdrm..
 */
static int get_plane_id(struct drm *drm)
{
	drmModePlaneResPtr plane_resources;
	uint32_t i, j;
	int ret = -EINVAL;
	int found_primary = 0;

	plane_resources = drmModeGetPlaneResources(drm->fd);
	if (!plane_resources) {
		printf("drmModeGetPlaneResources failed: %s\n", strerror(errno));
		return -1;
	}

	for (i = 0; (i < plane_resources->count_planes) && !found_primary; i++) {
		uint32_t id = plane_resources->planes[i];
		drmModePlanePtr plane = drmModeGetPlane(drm->fd, id);
		if (!plane) {
			printf("drmModeGetPlane(%u) failed: %s\n", id, strerror(errno));
			continue;
		}

		if (plane->possible_crtcs & (1 << drm->crtc_index)) {
			drmModeObjectPropertiesPtr props =
				drmModeObjectGetProperties(drm->fd, id, DRM_MODE_OBJECT_PLANE);

			/* primary or not, this plane is good enough to use: */
			ret = id;

			for (j = 0; j < props->count_props; j++) {
				drmModePropertyPtr p =
					drmModeGetProperty(drm->fd, props->props[j]);

				if ((strcmp(p->name, "type") == 0) &&
						(props->prop_values[j] == DRM_PLANE_TYPE_PRIMARY)) {
					/* found our primary plane, lets use that: */
					found_primary = 1;
				}

				drmModeFreeProperty(p);
			}

			drmModeFreeObjectProperties(props);
		}

		drmModeFreePlane(plane);
	}

	drmModeFreePlaneResources(plane_resources);

	return ret;
}

static int
get_properties(struct drm *drm, struct props *props,
	       const char *name, uint32_t type, uint32_t id)
{
	drmModeObjectProperties *drm_props;
	uint32_t i;

	drm_props = drmModeObjectGetProperties(drm->fd, id, type);
	if (!drm_props) {
		printf("could not get %s %u properties: %s\n",
		       name, id, strerror(errno));
		return -1;
	}

	props->count = drm_props->count_props;
	props->info = calloc(props->count, sizeof(props->info));
	for (i = 0; i < props->count; i++) {
		props->info[i] = drmModeGetProperty(drm->fd,
						    drm_props->props[i]);
	}

	drmModeFreeObjectProperties(drm_props);

	return 0;
}

static int
init_drm(struct drm *drm, const char *device)
{
	drmModeRes *resources;
	drmModeEncoder *encoder = NULL;
	int i, area, ret;
	uint32_t plane_id, crtc_id;

	drm->kms_in_fence_fd = -1,
	drm->kms_out_fence_fd = -1,
	drm->fd = open(device, O_RDWR);
	if (drm->fd < 0) {
		printf("could not open drm device\n");
		return -1;
	}

	ret = drmSetClientCap(drm->fd, DRM_CLIENT_CAP_ATOMIC, 1);
	if (ret) {
		printf("no atomic modesetting support: %s\n", strerror(errno));
		return -1;
	}

	resources = drmModeGetResources(drm->fd);
	if (!resources) {
		printf("drmModeGetResources failed: %s\n", strerror(errno));
		return -1;
	}

	/* find a connected connector: */
	for (i = 0; i < resources->count_connectors; i++) {
		drm->connector = drmModeGetConnector(drm->fd, resources->connectors[i]);
		if (drm->connector->connection == DRM_MODE_CONNECTED) {
			/* it's connected, let's use this! */
			break;
		}
		drmModeFreeConnector(drm->connector);
	}

	if (i == resources->count_connectors) {
		/* we could be fancy and listen for hotplug events and wait for
		 * a connector..
		 */
		printf("no connected connector!\n");
		return -1;
	}

	/* find preferred mode or the highest resolution mode: */
	for (i = 0, area = 0; i < drm->connector->count_modes; i++) {
		drmModeModeInfo *current_mode = &drm->connector->modes[i];

		if (current_mode->type & DRM_MODE_TYPE_PREFERRED) {
			drm->mode = current_mode;
		}

		int current_area = current_mode->hdisplay * current_mode->vdisplay;
		if (current_area > area) {
			drm->mode = current_mode;
			area = current_area;
		}
	}

	if (!drm->mode) {
		printf("could not find mode!\n");
		return -1;
	}

	/* find encoder: */
	for (i = 0; i < resources->count_encoders; i++) {
		encoder = drmModeGetEncoder(drm->fd, resources->encoders[i]);
		if (encoder->encoder_id == drm->connector->encoder_id)
			break;
		drmModeFreeEncoder(encoder);
		encoder = NULL;
	}

	if (encoder) {
		crtc_id = encoder->crtc_id;
	} else {
		crtc_id = find_crtc_for_connector(drm, resources, drm->connector);
		if (crtc_id == 0) {
			printf("no crtc found!\n");
			return -1;
		}
	}

	for (i = 0; i < resources->count_crtcs; i++) {
		if (resources->crtcs[i] == crtc_id) {
			drm->crtc_index = i;
			break;
		}
	}

	drm->crtc = drmModeGetCrtc(drm->fd, crtc_id);

	drmModeFreeResources(resources);

	ret = get_plane_id(drm);
	if (!ret) {
		printf("could not find a suitable plane\n");
		return -1;
	} else {
		plane_id = ret;
	}

	drm->plane = drmModeGetPlane(drm->fd, plane_id);

	get_properties(drm, &drm->plane_props, "plane",
		       DRM_MODE_OBJECT_PLANE, plane_id);
	get_properties(drm, &drm->crtc_props, "crtc",
		       DRM_MODE_OBJECT_CRTC, drm->crtc->crtc_id);
	get_properties(drm, &drm->connector_props, "connector",
		       DRM_MODE_OBJECT_CONNECTOR, drm->connector->connector_id);

	return 0;
}

static bool has_ext(const char *extension_list, const char *ext)
{
	const char *ptr = extension_list;
	int len = strlen(ext);

	if (ptr == NULL || *ptr == '\0')
		return false;

	while (true) {
		ptr = strstr(ptr, ext);
		if (!ptr)
			return false;

		if (ptr[len] == ' ' || ptr[len] == '\0')
			return true;

		ptr += len;
	}
}

static inline int __egl_check(void *ptr, const char *name)
{
	if (!ptr) {
		printf("no %s\n", name);
		return -1;
	}
	return 0;
}

#define egl_check(egl, name) __egl_check((egl)->name, #name)

static int
init_egl(struct egl *egl, struct gpu_scroller *scroller)
{
	EGLint major, minor, n;

	static const EGLint context_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 3,
		EGL_NONE
	};

	static const EGLint config_attribs[] = {
		EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_RED_SIZE, 1,
		EGL_GREEN_SIZE, 1,
		EGL_BLUE_SIZE, 1,
		EGL_ALPHA_SIZE, 0,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
		EGL_NONE
	};
	const char *egl_exts_client, *egl_exts_dpy, *gl_exts;

#define get_proc_client(ext, name) do { \
		if (has_ext(egl_exts_client, #ext)) \
			egl->name = (void *)eglGetProcAddress(#name); \
	} while (0)
#define get_proc_dpy(ext, name) do { \
		if (has_ext(egl_exts_dpy, #ext)) \
			egl->name = (void *)eglGetProcAddress(#name); \
	} while (0)

#define get_proc_gl(ext, name) do { \
		if (has_ext(gl_exts, #ext)) \
			egl->name = (void *)eglGetProcAddress(#name); \
	} while (0)

	egl_exts_client = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
	get_proc_client(EGL_EXT_platform_base, eglGetPlatformDisplayEXT);

	if (egl->eglGetPlatformDisplayEXT) {
		egl->display = egl->eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_KHR,
				scroller->dev, NULL);
	} else {
		egl->display = eglGetDisplay((void *)scroller->dev);
	}

	if (!eglInitialize(egl->display, &major, &minor)) {
		printf("failed to initialize\n");
		return -1;
	}

	egl_exts_dpy = eglQueryString(egl->display, EGL_EXTENSIONS);
	get_proc_dpy(EGL_KHR_image_base, eglCreateImageKHR);
	get_proc_dpy(EGL_KHR_image_base, eglDestroyImageKHR);
	get_proc_dpy(EGL_KHR_fence_sync, eglCreateSyncKHR);
	get_proc_dpy(EGL_KHR_fence_sync, eglDestroySyncKHR);
	get_proc_dpy(EGL_KHR_fence_sync, eglWaitSyncKHR);
	get_proc_dpy(EGL_KHR_fence_sync, eglClientWaitSyncKHR);
	get_proc_dpy(EGL_ANDROID_native_fence_sync, eglDupNativeFenceFDANDROID);

	printf("Using display %p with EGL version %d.%d\n",
			egl->display, major, minor);

	printf("===================================\n");
	printf("EGL information:\n");
	printf("  version: \"%s\"\n", eglQueryString(egl->display, EGL_VERSION));
	printf("  vendor: \"%s\"\n", eglQueryString(egl->display, EGL_VENDOR));
	printf("  client extensions: \"%s\"\n", egl_exts_client);
	printf("  display extensions: \"%s\"\n", egl_exts_dpy);
	printf("===================================\n");

	if (!eglBindAPI(EGL_OPENGL_ES_API)) {
		printf("failed to bind api EGL_OPENGL_ES_API\n");
		return -1;
	}

	if (!eglChooseConfig(egl->display, config_attribs, &egl->config, 1, &n) || n != 1) {
		printf("failed to choose config: %d\n", n);
		return -1;
	}

	egl->context = eglCreateContext(egl->display, egl->config,
			EGL_NO_CONTEXT, context_attribs);
	if (egl->context == NULL) {
		printf("failed to create context\n");
		return -1;
	}

	egl->surface = eglCreateWindowSurface(egl->display, egl->config,
			(EGLNativeWindowType)scroller->surface, NULL);
	if (egl->surface == EGL_NO_SURFACE) {
		printf("failed to create egl surface\n");
		return -1;
	}

	/* connect the context to the surface */
	eglMakeCurrent(egl->display, egl->surface, egl->surface, egl->context);

	gl_exts = (char *) glGetString(GL_EXTENSIONS);
	printf("OpenGL ES 2.x information:\n");
	printf("  version: \"%s\"\n", glGetString(GL_VERSION));
	printf("  shading language version: \"%s\"\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("  vendor: \"%s\"\n", glGetString(GL_VENDOR));
	printf("  renderer: \"%s\"\n", glGetString(GL_RENDERER));
	printf("  extensions: \"%s\"\n", gl_exts);
	printf("===================================\n");

	get_proc_gl(GL_OES_EGL_image, glEGLImageTargetTexture2DOES);

	if (egl_check(egl, eglDupNativeFenceFDANDROID) ||
	    egl_check(egl, eglCreateSyncKHR) ||
	    egl_check(egl, eglDestroySyncKHR) ||
	    egl_check(egl, eglWaitSyncKHR) ||
	    egl_check(egl, eglClientWaitSyncKHR) ||
	    egl_check(egl, eglCreateImageKHR) ||
	    egl_check(egl, glEGLImageTargetTexture2DOES) ||
	    egl_check(egl, eglDestroyImageKHR))
		return -1;

	return 0;
}

static int
create_program(const char *vs_src, const char *fs_src)
{
	GLuint vertex_shader, fragment_shader, program;
	GLint ret;

	vertex_shader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertex_shader, 1, &vs_src, NULL);
	glCompileShader(vertex_shader);

	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &ret);
	if (!ret) {
		char *log;

		printf("vertex shader compilation failed!:\n");
		glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &ret);
		if (ret > 1) {
			log = malloc(ret);
			glGetShaderInfoLog(vertex_shader, ret, NULL, log);
			printf("%s", log);
		}

		return -1;
	}

	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragment_shader, 1, &fs_src, NULL);
	glCompileShader(fragment_shader);

	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &ret);
	if (!ret) {
		char *log;

		printf("fragment shader compilation failed!:\n");
		glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &ret);

		if (ret > 1) {
			log = malloc(ret);
			glGetShaderInfoLog(fragment_shader, ret, NULL, log);
			printf("%s", log);
		}

		return -1;
	}

	program = glCreateProgram();

	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);

	return program;
}

static int
link_program(unsigned program)
{
	GLint ret;

	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &ret);
	if (!ret) {
		char *log;

		printf("program linking failed!:\n");
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &ret);

		if (ret > 1) {
			log = malloc(ret);
			glGetProgramInfoLog(program, ret, NULL, log);
			printf("%s", log);
		}

		return -1;
	}

	return 0;
}

static const char *vertex_shader_source =
	"#version 300 es\n"
	"in vec4 in_position;\n"
	"\n"
	"out vec2 vTexCoord;\n"
	"\n"
	"void main()\n"
	"{\n"
	"    gl_Position = in_position;\n"
	"    vec2 tex_coords[4] = vec2[] \n"
	"        (vec2(1.0, 1.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0));\n"
	"    vTexCoord = tex_coords[gl_VertexID];\n"
	"}\n";

static const char *fragment_shader_source = 
	"#version 300 es\n"
	"precision mediump float;\n"
	"\n"
	"uniform sampler2D uTex;\n"
	"\n"
	"in vec2 vTexCoord;\n"
	"out vec4 frag_color;\n"
	"\n"
	"void main()\n"
	"{\n"
	"    frag_color = texture(uTex, vTexCoord);\n"
	"}\n";

static int
init_gl(struct gpu_scroller *scroller)
{
	int ret;

	ret = init_egl(&scroller->egl, scroller);
	if (ret)
		return ret;

	ret = create_program(vertex_shader_source, fragment_shader_source);
	if (ret < 0)
		return ret;

	scroller->program = ret;

	glBindAttribLocation(scroller->program, 0, "in_position");
	glBindAttribLocation(scroller->program, 1, "in_TexCoord");

	ret = link_program(scroller->program);
	if (ret)
		return ret;

	glUseProgram(scroller->program);

	scroller->texture = glGetUniformLocation(scroller->program, "uTex");

	glViewport(0, 0, scroller->width, scroller->height);

	int drm_fd = gbm_device_get_fd(scroller->dev);
	for (uint32_t i = 0; i < ARRAY_SIZE(scroller->tiles); i++) {
		scroller->tiles[i] = create_tile(drm_fd, tile_width, tile_height);
		if (scroller->tiles[i] == NULL) {
			printf("failed to initialize EGLImage texture\n");
			return -1;
		}

		if (create_texture_for_tile(drm_fd, &scroller->egl,
					    scroller->tiles[i],
					    tile_width, tile_height) < 0)
			return -1;
	}

	glGenBuffers(1, &scroller->vbo);
	glBindBuffer(GL_ARRAY_BUFFER, scroller->vbo);
	glBufferData(GL_ARRAY_BUFFER, 4096, 0, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (const GLvoid *)(intptr_t)0);
	glEnableVertexAttribArray(0);

	return 0;
}

static void draw_cube_tex(struct gpu_scroller *scroller,
			  uint32_t offset_x, uint32_t offset_y)
{
	glClearColor(0.5, 0.5, 0.5, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glUniform1i(scroller->texture, 0);

	uint32_t tile_x = offset_x / tile_width;
	uint32_t tile_offset_x = offset_x & (tile_width - 1);
	uint32_t tile_y = offset_y / tile_height;
	uint32_t tile_offset_y = offset_y & (tile_height - 1);

	const float fw = 2 * tile_width / (float) scroller->width;
	const float fh = 2 * tile_height / (float) scroller->height;

	const float start_fx = -1.0f - 2 * tile_offset_x / (float) scroller->width;
	const float start_fy =  1.0f + 2 * tile_offset_y / (float) scroller->height;

	uint32_t tile_row = tile_y;

	for (float fy = start_fy; fy > -1.0f; fy -= fh) {
		uint32_t tile_index = tile_row * tile_row_stride + tile_x;
		tile_row++;

		for (float fx = start_fx; fx < 1.0f; fx += fw) {

			const GLfloat vertices[] = {
				fx, fy,
				fx + fw, fy,
				fx,  fy - fh,
				fx + fw, fy - fh,
			};

			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), &vertices[0]);
			glBindTexture(GL_TEXTURE_2D, scroller->tiles[tile_index & (ARRAY_SIZE(scroller->tiles) - 1)]->texture);

			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

			tile_index++;
		}
	}
}

static int
gpu_scroller_draw(struct scroller *base_scroller, uint32_t offset_x, uint32_t offset_y)
{
	struct gpu_scroller *scroller = container_of(base_scroller, struct gpu_scroller, base);
	struct drm_fb *fb;
	struct egl *egl = &scroller->egl;
	uint32_t flags = DRM_MODE_ATOMIC_NONBLOCK;
	int ret;

	/* Allow a modeset change for the first commit only. */
	flags |= DRM_MODE_ATOMIC_ALLOW_MODESET;

	struct gbm_bo *next_bo;
	EGLSyncKHR gpu_fence = NULL;   /* out-fence from gpu, in-fence to kms */
	EGLSyncKHR kms_fence = NULL;   /* in-fence to gpu, out-fence from kms */

	if (scroller->drm.kms_out_fence_fd != -1) {
		kms_fence = create_fence(egl, scroller->drm.kms_out_fence_fd);
		assert(kms_fence);

		/* driver now has ownership of the fence fd: */
		scroller->drm.kms_out_fence_fd = -1;

		/* wait "on the gpu" (ie. this won't necessarily block, but
		 * will block the rendering until fence is signaled), until
		 * the previous pageflip completes so we don't render into
		 * the buffer that is still on screen.
		 */
		egl->eglWaitSyncKHR(egl->display, kms_fence, 0);
	}

	draw_cube_tex(scroller, offset_x, offset_y);

	/* insert fence to be singled in cmdstream.. this fence will be
	 * signaled when gpu rendering done
	 */
	gpu_fence = create_fence(egl, EGL_NO_NATIVE_FENCE_FD_ANDROID);
	assert(gpu_fence);

	eglSwapBuffers(egl->display, egl->surface);

	/* after swapbuffers, gpu_fence should be flushed, so safe
	 * to get fd:
	 */
	scroller->drm.kms_in_fence_fd = egl->eglDupNativeFenceFDANDROID(egl->display, gpu_fence);
	egl->eglDestroySyncKHR(egl->display, gpu_fence);
	assert(scroller->drm.kms_in_fence_fd != -1);

	next_bo = gbm_surface_lock_front_buffer(scroller->surface);
	if (!next_bo) {
		printf("Failed to lock frontbuffer\n");
		return -1;
	}
	fb = drm_fb_get_from_bo(scroller, next_bo);
	if (!fb) {
		printf("Failed to get a new framebuffer BO\n");
		return -1;
	}

	if (kms_fence) {
		EGLint status;

		/* Wait on the CPU side for the _previous_ commit to
		 * complete before we post the flip through KMS, as
		 * atomic will reject the commit if we post a new one
		 * whilst the previous one is still pending.
		 */
		do {
			status = egl->eglClientWaitSyncKHR(egl->display,
							   kms_fence,
							   0,
							   EGL_FOREVER_KHR);
		} while (status != EGL_CONDITION_SATISFIED_KHR);

		egl->eglDestroySyncKHR(egl->display, kms_fence);
	}

	/*
	 * Here you could also update drm plane layers if you want
	 * hw composition
	 */
	ret = drm_atomic_commit(&scroller->drm, fb->fb_id,
				0, 0, scroller->width, scroller->height, flags);
	if (ret) {
		printf("failed to commit: %s\n", strerror(errno));
		return -1;
	}

	/* release last buffer to render on again: */
	if (scroller->last_bo)
		gbm_surface_release_buffer(scroller->surface, scroller->last_bo);
	scroller->last_bo = next_bo;

	return 0;
}

static struct scroller *
create_gpu_scroller(const char *device)
{
	struct gpu_scroller *scroller = malloc(sizeof(*scroller));

	scroller->last_bo = NULL;
	if (init_drm(&scroller->drm, device) < 0) {
		printf("failed to initialize atomic DRM\n");
		return NULL;
	}

	scroller->dev = gbm_create_device(scroller->drm.fd);
	scroller->width = scroller->drm.mode->hdisplay - 200;
	scroller->height = scroller->drm.mode->vdisplay - 200;

	static const uint64_t modifiers[] = { I915_FORMAT_MOD_Y_TILED };
	scroller->surface = gbm_surface_create_with_modifiers(scroller->dev,
							      scroller->width,
							      scroller->height,
							      GBM_FORMAT_ABGR8888,
							      modifiers, ARRAY_SIZE(modifiers));
	if (!scroller->surface) {
		printf("failed to create gbm surface\n");
		return NULL;
	}

	if (init_gl(scroller) < 0) {
		printf("failed to initialize gl\n");
		return NULL;
	}

	scroller->base.draw = gpu_scroller_draw;

	return &scroller->base;
}

struct plane_scroller {
	struct scroller base;

	struct drm drm;
	int width, height;

	struct scratch_bo *scratch_bo;
	struct tile *tiles[64];
};

static struct scratch_bo *
create_scratch_bo(struct plane_scroller *scroller)
{
	struct scratch_bo *sbo = malloc(sizeof(*sbo));
	int fd = scroller->drm.fd;

	uint32_t scratch_width = 4096;
	uint32_t scratch_height = 4096;
	sbo->stride = scratch_width * 4;
	uint32_t scratch_size = sbo->stride * scratch_height;
	struct drm_i915_gem_create_v2 scratch_create = {
		.size = scratch_size,
		.flags = I915_GEM_CREATE_SCRATCH
	};

	int ret = safe_ioctl(fd, DRM_IOCTL_I915_GEM_CREATE, &scratch_create);
	if (ret < 0) {
		fprintf(stderr, "scratch gem create v2 failed: %s\n", strerror(errno));
		return NULL;
	}

	sbo->gem_handle = scratch_create.handle;

	uint32_t tile_stride = scroller->tiles[0]->stride;
	for (uint32_t y = 0; y < scratch_height / tile_height; y++) {
		for (uint32_t x = 0; x < sbo->stride / tile_stride; x++) {
			uint32_t tile = (y * tile_row_stride + x) % ARRAY_SIZE(scroller->tiles);
			/* All offsets, stride, width and height are in pages */
			struct drm_i915_gem_set_pages set_pages = {
				.dst_handle = scratch_create.handle,
				.src_handle = scroller->tiles[tile]->gem_handle,

				.dst_offset = x * tile_stride / 128 + y * sbo->stride / 128 * tile_height / 32,
				.src_offset = 0,

				/* A Y tile is 128 bytes wide, so divide the stride by
				 * 128 to find the number of tiles (ie pages). */
				.dst_stride = sbo->stride / 128,
				.src_stride = tile_stride / 128,

				/* Again, divide stride by 128 to get the width of the
				 * region in tiles. Y tiles are 32 lines high, so
				 * divide height by 32 to get height in number of
				 * pages. */
				.width = tile_stride / 128,
				.height = tile_height / 32,
			};

			ret = safe_ioctl(fd, DRM_IOCTL_I915_GEM_SET_PAGES, &set_pages);
			if (ret < 0) {
				fprintf(stderr, "tile gem create v2 failed: %s\n", strerror(errno));
				return NULL;
			}
		}
	}

	uint32_t handles[4] = { sbo->gem_handle, };
	uint32_t strides[4] = { sbo->stride, };
	uint32_t offsets[4] = { 0, };
	uint64_t modifiers[4] = { I915_FORMAT_MOD_Y_TILED, };

	ret = drmModeAddFB2WithModifiers(fd, scratch_width, scratch_height,
					 DRM_FORMAT_XRGB8888, handles, strides, offsets,
					 modifiers, &sbo->fb_id, DRM_MODE_FB_MODIFIERS);

	if (ret) {
		printf("failed to create fb: %s\n", strerror(errno));
		free(sbo);
		return NULL;
	}

	return sbo;
}

static int
plane_scroller_draw(struct scroller *base_scroller, uint32_t offset_x, uint32_t offset_y)
{
	struct plane_scroller *scroller = container_of(base_scroller, struct plane_scroller, base);
	uint32_t flags = DRM_MODE_ATOMIC_ALLOW_MODESET;

	int ret;

	ret = drm_atomic_commit(&scroller->drm, scroller->scratch_bo->fb_id,
				offset_x, offset_y,
				scroller->width, scroller->height, flags);
	if (ret) {
		printf("failed to commit: %s\n", strerror(errno));
		return -1;
	}

	return 0;
}

static struct scroller *
create_plane_scroller(const char *device)
{
	struct plane_scroller *scroller = malloc(sizeof(*scroller));

	if (init_drm(&scroller->drm, device) < 0) {
		printf("failed to initialize atomic DRM\n");
		return NULL;
	}

	scroller->width = scroller->drm.mode->hdisplay - 200;
	scroller->height = scroller->drm.mode->vdisplay - 200;

	for (uint32_t i = 0; i < ARRAY_SIZE(scroller->tiles); i++) {
		scroller->tiles[i] = create_tile(scroller->drm.fd,
						 tile_width, tile_height);
		if (scroller->tiles[i] == NULL) {
			printf("failed to initialize EGLImage texture\n");
			return NULL;
		}
	}

	scroller->scratch_bo = create_scratch_bo(scroller);

	scroller->base.draw = plane_scroller_draw;

	return &scroller->base;
}

static const char *shortopts = "D:M:";

static const struct option longopts[] = {
	{"device", required_argument, 0, 'D'},
	{"mode",   required_argument, 0, 'M'},
	{0, 0, 0, 0}
};

static void usage(const char *name)
{
	printf("Usage: %s [-DM]\n"
	       "\n"
	       "options:\n"
	       "    -D, --device=DEVICE      use the given device\n"
	       "    -M, --mode=MODE          specify mode, one of:\n"
	       "        gpu       -  gpu rendered scrolling (default)\n"
	       "        plane     -  plane rendered scrolling\n",
	       name);
}

static struct termios save_tio;

static void restore_vt(void)
{
	struct vt_mode mode = { .mode = VT_AUTO };
	ioctl(STDIN_FILENO, VT_SETMODE, &mode);

	tcsetattr(STDIN_FILENO, TCSANOW, &save_tio);
	ioctl(STDIN_FILENO, KDSETMODE, KD_TEXT);
}

static void handle_signal(int sig)
{
	restore_vt();

	raise(sig);
}

static int init_vt(void)
{
	struct termios tio;
	struct stat buf;
	int ret;

	/* If we're not on a VT, we're probably logged in as root over
	 * ssh. Skip all this then. */
	ret = fstat(STDIN_FILENO, &buf);
	if (ret == -1 || major(buf.st_rdev) != TTY_MAJOR)
		return 0;

	/* First, save term io setting so we can restore properly. */
	tcgetattr(STDIN_FILENO, &save_tio);

	/* We don't drop drm master, so block VT switching while we're
	 * running. Otherwise, switching to X on another VT will crash X when it
	 * fails to get drm master. */
	struct vt_mode mode = { .mode = VT_PROCESS, .relsig = 0, .acqsig = 0 };
	ret = ioctl(STDIN_FILENO, VT_SETMODE, &mode);
	if (ret == -1) {
		printf("failed to take control of vt handling\n");
		return -1;
	}

	/* Set KD_GRAPHICS to disable fbcon while we render. */
	ret = ioctl(STDIN_FILENO, KDSETMODE, KD_GRAPHICS);
	if (ret == -1) {
		printf("failed to switch console to graphics mode\n");
		return -1;
	}

	atexit(restore_vt);

	/* Set console input to raw mode. */
	tio = save_tio;
	tio.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &tio);

	/* Restore console on SIGINT and friends. */
	struct sigaction act = {
		.sa_handler = handle_signal,
		.sa_flags = SA_RESETHAND
	};
	sigaction(SIGINT, &act, NULL);
	sigaction(SIGSEGV, &act, NULL);
	sigaction(SIGABRT, &act, NULL);

	return 0;
}

int main(int argc, char *argv[])
{
	const char *device = "/dev/dri/card0";
	int opt;
	struct scroller *scroller;
	enum { MODE_GPU, MODE_PLANE } mode = MODE_GPU;

	while ((opt = getopt_long_only(argc, argv, shortopts, longopts, NULL)) != -1) {
		switch (opt) {
		case 'D':
			device = optarg;
			break;
		case 'M':
			if (strcmp(optarg, "gpu") == 0) {
				mode = MODE_GPU;
			} else if (strcmp(optarg, "plane") == 0) {
				mode = MODE_PLANE;
			} else {
				printf("invalid mode: %s\n", optarg);
				usage(argv[0]);
				return -1;
			}
			break;
		default:
			usage(argv[0]);
			return -1;
		}
	}

	switch (mode) {
	case MODE_GPU:
		scroller = create_gpu_scroller(device);
		break;
	case MODE_PLANE:
		scroller = create_plane_scroller(device);
		break;
	}

	if (!scroller) {
		printf("failed to initialize scroller\n");
		return -1;
	}

	if (init_vt())
		return -1;

	uint32_t i;
	struct pollfd pfd[1] = { { .fd = 0, .events = POLLIN } };
	int ret = 0;
	while (poll(pfd, 1, 0) == 0 && ret == 0) {
		const uint32_t offset_x = cos((i & 255) / 256.0f * 2 * M_PI) * 150 + 200;
		const uint32_t offset_y = sin((i & 255) / 256.0f * 2 * M_PI) * 150 + 200;
		i++;

		ret = scroller->draw(scroller, offset_x, offset_y);
	}

	return 0;
}
