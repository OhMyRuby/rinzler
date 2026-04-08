/*
 * vulkan_gemm.c — GPU matrix multiply via Vulkan compute
 *
 * Exposes Rinzler::Vulkan.gemm(a, b, m, k, n) → flat Float array
 *
 * Lifecycle:
 *   - On require: initialize Vulkan instance, pick device, create command pool
 *   - On each gemm call: allocate buffers, upload, dispatch shader, download, free
 *   - On GC/exit: destroy Vulkan context
 *
 * We use host-visible, host-coherent buffers for simplicity. For large matrices
 * device-local buffers + staging would be faster, but the transfer overhead
 * dominates only for very large matrices — fine to optimize later.
 */

#include "ruby.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── Helpers ──────────────────────────────────────────────────────────────── */

#define VK_CHECK(call)                                                          \
    do {                                                                        \
        VkResult _r = (call);                                                   \
        if (_r != VK_SUCCESS) {                                                 \
            rb_raise(rb_eRuntimeError, "Vulkan error %d at %s:%d",             \
                     _r, __FILE__, __LINE__);                                   \
        }                                                                       \
    } while (0)

/* ── Global Vulkan context (initialized once at load time) ────────────────── */

typedef struct {
    VkInstance               instance;
    VkPhysicalDevice         physical_device;
    VkDevice                 device;
    uint32_t                 compute_queue_family;
    VkQueue                  compute_queue;
    VkCommandPool            command_pool;
    VkDescriptorSetLayout    descriptor_set_layout;
    VkPipelineLayout         pipeline_layout;
    VkPipeline               compute_pipeline;
    VkDescriptorPool         descriptor_pool;
    int                      initialized;
} VulkanCtx;

static VulkanCtx ctx = {0};

/* ── SPIR-V shader loading ────────────────────────────────────────────────── */

static uint32_t *load_spirv(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) rb_raise(rb_eRuntimeError, "Cannot open shader: %s", path);

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint32_t *buf = malloc(size);
    if (!buf) { fclose(f); rb_raise(rb_eNoMemError, "OOM loading shader"); }

    fread(buf, 1, size, f);
    fclose(f);

    *out_size = size;
    return buf;
}

/* ── Buffer helpers ───────────────────────────────────────────────────────── */

static void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                           VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo bi = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(ctx.device, &bi, NULL, buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx.device, *buf, &mr);

    /* Find host-visible, host-coherent memory */
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &mp);

    uint32_t mem_type = UINT32_MAX;
    VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & wanted) == wanted) {
            mem_type = i;
            break;
        }
    }
    if (mem_type == UINT32_MAX)
        rb_raise(rb_eRuntimeError, "No suitable Vulkan memory type found");

    VkMemoryAllocateInfo ai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = mr.size,
        .memoryTypeIndex = mem_type,
    };
    VK_CHECK(vkAllocateMemory(ctx.device, &ai, NULL, mem));
    VK_CHECK(vkBindBufferMemory(ctx.device, *buf, *mem, 0));
}

/* ── Vulkan initialization ────────────────────────────────────────────────── */

static void vulkan_init(const char *shader_path) {
    /* Instance */
    VkApplicationInfo app = {
        .sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = "rinzler-vulkan",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion         = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo ici = {
        .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VK_CHECK(vkCreateInstance(&ici, NULL, &ctx.instance));

    /* Physical device — pick the first one */
    uint32_t n_dev = 1;
    VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &n_dev, &ctx.physical_device));
    if (n_dev == 0) rb_raise(rb_eRuntimeError, "No Vulkan physical devices found");

    /* Find a compute queue family */
    uint32_t n_qf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device, &n_qf, NULL);
    VkQueueFamilyProperties *qf = alloca(n_qf * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device, &n_qf, qf);

    ctx.compute_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < n_qf; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx.compute_queue_family = i;
            break;
        }
    }
    if (ctx.compute_queue_family == UINT32_MAX)
        rb_raise(rb_eRuntimeError, "No compute queue family found");

    /* Logical device */
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_queue_family,
        .queueCount       = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos    = &qci,
    };
    VK_CHECK(vkCreateDevice(ctx.physical_device, &dci, NULL, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.compute_queue_family, 0, &ctx.compute_queue);

    /* Command pool */
    VkCommandPoolCreateInfo cpci = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_queue_family,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    VK_CHECK(vkCreateCommandPool(ctx.device, &cpci, NULL, &ctx.command_pool));

    /* Descriptor set layout: 3 storage buffers (A, B, C) */
    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < 3; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding         = i,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings    = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dslci, NULL,
                                         &ctx.descriptor_set_layout));

    /* Pipeline layout with push constants for M, K, N */
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 3 * sizeof(uint32_t),
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &ctx.descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pcr,
    };
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &plci, NULL, &ctx.pipeline_layout));

    /* Compute pipeline from SPIR-V */
    size_t   spv_size;
    uint32_t *spv = load_spirv(shader_path, &spv_size);

    VkShaderModuleCreateInfo smci = {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_size,
        .pCode    = spv,
    };
    VkShaderModule shader_module;
    VkResult r = vkCreateShaderModule(ctx.device, &smci, NULL, &shader_module);
    free(spv);
    VK_CHECK(r);

    VkComputePipelineCreateInfo cpci2 = {
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName  = "main",
        },
        .layout = ctx.pipeline_layout,
    };
    VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1,
                                       &cpci2, NULL, &ctx.compute_pipeline));
    vkDestroyShaderModule(ctx.device, shader_module, NULL);

    /* Descriptor pool */
    /* 3 bindings per set × up to 8 concurrent sets */
    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 24,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        /* FREE_DESCRIPTOR_SET_BIT allows vkFreeDescriptorSets per call.
           maxSets = 8 gives headroom for concurrent/batched dispatches. */
        .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets       = 8,
        .poolSizeCount = 1,
        .pPoolSizes    = &pool_size,
    };
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &dpci, NULL, &ctx.descriptor_pool));

    ctx.initialized = 1;
}

/* ── Vulkan teardown ──────────────────────────────────────────────────────── */

static void vulkan_cleanup(void) {
    if (!ctx.initialized) return;
    vkDestroyDescriptorPool(ctx.device, ctx.descriptor_pool, NULL);
    vkDestroyPipeline(ctx.device, ctx.compute_pipeline, NULL);
    vkDestroyPipelineLayout(ctx.device, ctx.pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx.device, ctx.descriptor_set_layout, NULL);
    vkDestroyCommandPool(ctx.device, ctx.command_pool, NULL);
    vkDestroyDevice(ctx.device, NULL);
    vkDestroyInstance(ctx.instance, NULL);
    ctx.initialized = 0;
}

/* ── Ruby binding: Rinzler::Vulkan.gemm(a, b, m, k, n) ───────────────────── */

static VALUE rb_vulkan_gemm(VALUE self, VALUE rb_a, VALUE rb_b,
                             VALUE rb_m, VALUE rb_k, VALUE rb_n) {
    if (!ctx.initialized)
        rb_raise(rb_eRuntimeError, "Vulkan not initialized — call Rinzler::Vulkan.init first");

    uint32_t M = NUM2UINT(rb_m);
    uint32_t K = NUM2UINT(rb_k);
    uint32_t N = NUM2UINT(rb_n);

    size_t bytes_a = M * K * sizeof(float);
    size_t bytes_b = K * N * sizeof(float);
    size_t bytes_c = M * N * sizeof(float);

    /* ── Allocate buffers ── */
    VkBuffer     buf_a, buf_b, buf_c;
    VkDeviceMemory mem_a, mem_b, mem_c;

    create_buffer(bytes_a, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_a, &mem_a);
    create_buffer(bytes_b, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_b, &mem_b);
    create_buffer(bytes_c, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_c, &mem_c);

    /* ── Upload A ── */
    void *mapped;
    vkMapMemory(ctx.device, mem_a, 0, bytes_a, 0, &mapped);
    for (uint32_t i = 0; i < M * K; i++) {
        ((float *)mapped)[i] = (float)NUM2DBL(rb_ary_entry(rb_a, i));
    }
    vkUnmapMemory(ctx.device, mem_a);

    /* ── Upload B ── */
    vkMapMemory(ctx.device, mem_b, 0, bytes_b, 0, &mapped);
    for (uint32_t i = 0; i < K * N; i++) {
        ((float *)mapped)[i] = (float)NUM2DBL(rb_ary_entry(rb_b, i));
    }
    vkUnmapMemory(ctx.device, mem_b);

    /* ── Descriptor set ── */
    VkDescriptorSetAllocateInfo dsai = {
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = ctx.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &ctx.descriptor_set_layout,
    };
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &dsai, &ds));

    VkDescriptorBufferInfo buf_info[3] = {
        { buf_a, 0, bytes_a },
        { buf_b, 0, bytes_b },
        { buf_c, 0, bytes_c },
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = ds,
            .dstBinding      = i,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo     = &buf_info[i],
        };
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

    /* ── Record & submit command buffer ── */
    VkCommandBufferAllocateInfo cbai = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = ctx.command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cbai, &cb));

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.compute_pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx.pipeline_layout, 0, 1, &ds, 0, NULL);

    uint32_t push[3] = { M, K, N };
    vkCmdPushConstants(cb, ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    /* Dispatch: one workgroup per 16×16 tile of C */
    uint32_t gx = (N + 15) / 16;
    uint32_t gy = (M + 15) / 16;
    vkCmdDispatch(cb, gx, gy, 1);

    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cb,
    };

    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(ctx.device, &fci, NULL, &fence));
    VK_CHECK(vkQueueSubmit(ctx.compute_queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX));

    /* ── Read back C ── */
    vkMapMemory(ctx.device, mem_c, 0, bytes_c, 0, &mapped);
    VALUE result = rb_ary_new_capa(M * N);
    for (uint32_t i = 0; i < M * N; i++) {
        rb_ary_push(result, DBL2NUM(((float *)mapped)[i]));
    }
    vkUnmapMemory(ctx.device, mem_c);

    /* ── Cleanup ── */
    vkDestroyFence(ctx.device, fence, NULL);
    vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cb);
    vkFreeDescriptorSets(ctx.device, ctx.descriptor_pool, 1, &ds);
    vkDestroyBuffer(ctx.device, buf_c, NULL); vkFreeMemory(ctx.device, mem_c, NULL);
    vkDestroyBuffer(ctx.device, buf_b, NULL); vkFreeMemory(ctx.device, mem_b, NULL);
    vkDestroyBuffer(ctx.device, buf_a, NULL); vkFreeMemory(ctx.device, mem_a, NULL);

    return result;
}

/* ── Ruby binding: Rinzler::Vulkan.init(shader_path) ─────────────────────── */

static VALUE rb_vulkan_init(VALUE self, VALUE shader_path) {
    if (ctx.initialized) return Qtrue;
    vulkan_init(StringValueCStr(shader_path));
    return Qtrue;
}

static VALUE rb_vulkan_initialized_p(VALUE self) {
    return ctx.initialized ? Qtrue : Qfalse;
}

/* ── Extension entry point ────────────────────────────────────────────────── */

void Init_vulkan_ext(void) {
    VALUE rb_mRinzler = rb_define_module("Rinzler");
    VALUE mVulkan     = rb_define_module_under(rb_mRinzler, "Vulkan");

    rb_define_singleton_method(mVulkan, "init",          rb_vulkan_init,          1);
    rb_define_singleton_method(mVulkan, "initialized?",  rb_vulkan_initialized_p, 0);
    rb_define_singleton_method(mVulkan, "gemm",          rb_vulkan_gemm,          5);

    /* Clean up Vulkan on Ruby exit */
    rb_set_end_proc((void(*)(VALUE))vulkan_cleanup, Qnil);
}
