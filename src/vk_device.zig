const std = @import("std");
const FixedBufferAllocator = std.heap.FixedBufferAllocator;
const FixedArray = @import("fixed_array.zig").FixedArray;
const c = @import("c.zig").c;
const util = @import("util.zig");
const logger = @import("logger.zig");
const vk_shared = @import("vk_shared.zig");
const vk_check = vk_shared.vk_check;

const QueueFamilyIndices = struct {
    graphics: u8,
    present: u8,
    transfer: u8,
    compute: u8,
};

const SwapchainSupportInfo = struct {
    formats: FixedArray(c.VkSurfaceFormatKHR, 128),
    present_modes: FixedArray(c.VkPresentModeKHR, 8),
    capabilities: c.VkSurfaceCapabilitiesKHR,
};

pub const VulkanDevice = struct {
    swapchain_support_info: SwapchainSupportInfo,
    physical_device: c.VkPhysicalDevice,
    logical_device: c.VkDevice,
    memory_properties: c.VkPhysicalDeviceMemoryProperties,
    graphics_queue: c.VkQueue,
    present_queue: c.VkQueue,
    transfer_queue: c.VkQueue,
    depth_format: c.VkFormat,
    queue_family_indices: QueueFamilyIndices,

    const Self = @This();
    const MAX_QUEUE_COUNT = 4;

    pub fn init(instance: c.VkInstance, surface: c.VkSurfaceKHR, out_device: *VulkanDevice) bool {
        var device_cnt: u32 = undefined;
        vk_check(c.vkEnumeratePhysicalDevices.?(instance, &device_cnt, null));
        if (device_cnt == 0) {
            return false;
        }

        const MAX_DEVICES = 10;
        std.debug.assert(MAX_DEVICES >= device_cnt);

        var phys_devices: FixedArray(c.VkPhysicalDevice, MAX_DEVICES) = .init_len(device_cnt);
        vk_check(c.vkEnumeratePhysicalDevices.?(instance, &device_cnt, phys_devices.data()));

        const required_device_extensions: FixedArray([*c]const u8, 3) = .init_from_slice(&[_][*c]const u8{
            @ptrCast(c.VK_KHR_SWAPCHAIN_EXTENSION_NAME),
            @ptrCast(c.VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME),
            @ptrCast(c.VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME),
        });

        var found = false;
        for (phys_devices.slice_full()) |phys_device| {
            if (is_device_suitable(phys_device, surface, required_device_extensions.slice_full_as_const(), &out_device.swapchain_support_info)) {
                out_device.physical_device = phys_device;
                found = true;
            }
        }

        if (!found) {
            return false;
        }

        if (!out_device.detect_depth_format()) {
            return false;
        }

        if (!find_queue_families(out_device.physical_device, &out_device.queue_family_indices, surface)) {
            return false;
        }

        c.vkGetPhysicalDeviceMemoryProperties.?(out_device.physical_device, &out_device.memory_properties);

        // Do not create additional queues for shared indices.
        var unique_queue_creation_indices: FixedArray(u32, MAX_QUEUE_COUNT) = .empty;
        retrieve_unique_queue_creation_indices(out_device.queue_family_indices, &unique_queue_creation_indices);
        util.assert_msg(MAX_QUEUE_COUNT >= unique_queue_creation_indices.len, "Should not exceed max capacity", .{});

        var queue_create_infos: FixedArray(c.VkDeviceQueueCreateInfo, MAX_QUEUE_COUNT) = .empty;
        queue_create_infos.resize(unique_queue_creation_indices.len);

        const queue_priorities = [MAX_QUEUE_COUNT]f32{ 1.0, 1.0, 1.0, 1.0 };

        for (queue_create_infos.slice_full(), 0..queue_create_infos.len) |*create_info, i| {
            create_info.sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            create_info.pNext = null;
            create_info.queueFamilyIndex = unique_queue_creation_indices.items[i];
            create_info.queueCount = 1;
            create_info.flags = 0;
            create_info.pQueuePriorities = &queue_priorities[i];
        }

        var device_features: c.VkPhysicalDeviceFeatures = .{
            .samplerAnisotropy = c.VK_TRUE,
        };

        var device_features13: c.VkPhysicalDeviceVulkan13Features = .{
            .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .dynamicRendering = c.VK_TRUE,
            .synchronization2 = c.VK_TRUE,
        };

        var device_features12: c.VkPhysicalDeviceVulkan12Features = .{
            .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &device_features13,
            .bufferDeviceAddress = c.VK_TRUE,
            .descriptorIndexing = c.VK_TRUE,
        };

        const device_create_info: c.VkDeviceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &device_features12,
            .queueCreateInfoCount = queue_create_infos.len,
            .pQueueCreateInfos = queue_create_infos.data(),
            .enabledExtensionCount = required_device_extensions.len,
            .ppEnabledExtensionNames = required_device_extensions.data_as_const(),
            .pEnabledFeatures = &device_features,
        };

        // Create the device.
        vk_check(c.vkCreateDevice.?(out_device.physical_device, &device_create_info, null, &out_device.logical_device));
        c.volkLoadDevice(out_device.logical_device);

        // Get queues.
        if (out_device.queue_family_indices.graphics == out_device.queue_family_indices.present) {
            var shared_queue: c.VkQueue = undefined;
            c.vkGetDeviceQueue.?(out_device.logical_device, out_device.queue_family_indices.graphics, 0, &shared_queue);
            out_device.graphics_queue = shared_queue;
            out_device.present_queue = shared_queue;
        } else {
            c.vkGetDeviceQueue.?(out_device.logical_device, out_device.queue_family_indices.graphics, 0, &out_device.graphics_queue);
            c.vkGetDeviceQueue.?(out_device.logical_device, out_device.queue_family_indices.present, 0, &out_device.present_queue);
        }
        c.vkGetDeviceQueue.?(out_device.logical_device, out_device.queue_family_indices.transfer, 0, &out_device.transfer_queue);

        return true;
    }

    pub fn find_memory_index(self: *const VulkanDevice, type_filter: u32, memory_flags: c.VkMemoryPropertyFlagBits) ?u32 {
        for (self.memory_properties.memoryTypes[0..self.memory_properties.memoryTypeCount], 0..self.memory_properties.memoryTypeCount) |memory_type, i| {
            const shift = @as(u64, 1) << @as(u6, @intCast(i));
            if (((type_filter & shift) > 0) and ((memory_type.propertyFlags & memory_flags) == memory_flags)) {
                return @intCast(i);
            }
        }
        return null;
    }

    fn detect_depth_format(self: *Self) bool {
        const candidates = [_]c.VkFormat{
            c.VK_FORMAT_D32_SFLOAT,
            c.VK_FORMAT_D32_SFLOAT_S8_UINT,
            c.VK_FORMAT_D24_UNORM_S8_UINT,
        };

        const flags = c.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;

        for (candidates[0..]) |candidate| {
            var properties: c.VkFormatProperties = undefined;
            c.vkGetPhysicalDeviceFormatProperties.?(self.physical_device, candidate, &properties);
            if (((properties.linearTilingFeatures & flags) == flags) or ((properties.optimalTilingFeatures & flags) == flags)) {
                self.depth_format = candidate;
                return true;
            }
        }

        return false;
    }

    pub fn deinit(self: *Self) void {
        if (self.logical_device != null) {
            c.vkDestroyDevice.?(self.logical_device, null);
            self.logical_device = null;
        }
    }

    fn is_device_suitable(phys_device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR, required_extensions: []const [*c]const u8, out_swapchain_support_info: *SwapchainSupportInfo) bool {
        var physical_device_properties: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties.?(phys_device, &physical_device_properties);

        if (physical_device_properties.deviceType != c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            return false;
        }

        query_swapchain_support(phys_device, surface, out_swapchain_support_info);

        if (out_swapchain_support_info.present_modes.len == 0) {
            return false;
        }

        var avail_extension_count: u32 = undefined;
        vk_check(c.vkEnumerateDeviceExtensionProperties.?(phys_device, null, &avail_extension_count, null));

        const MAX_EXTENSIONS = 256;
        std.debug.assert(MAX_EXTENSIONS >= avail_extension_count);

        var avail_extensions: FixedArray(c.VkExtensionProperties, MAX_EXTENSIONS) = .init_len(avail_extension_count);
        vk_check(c.vkEnumerateDeviceExtensionProperties.?(phys_device, null, &avail_extension_count, avail_extensions.data()));

        outer: for (required_extensions) |req_ext| {
            for (avail_extensions.slice_full()) |avail_ext| {
                const req_ext_slice = std.mem.span(req_ext);
                if (std.mem.eql(u8, req_ext_slice, avail_ext.extensionName[0..req_ext_slice.len])) {
                    continue :outer;
                }
            }
            // required extension was not found
            return false;
        }

        var physical_device_features: c.VkPhysicalDeviceFeatures = undefined;
        c.vkGetPhysicalDeviceFeatures.?(phys_device, &physical_device_features);

        if (physical_device_features.samplerAnisotropy == 0) {
            return false;
        }

        return true;
    }

    fn retrieve_unique_queue_creation_indices(queue_indices: QueueFamilyIndices, out_unique_indices: *FixedArray(u32, MAX_QUEUE_COUNT)) void {
        var index_map: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty;
        var buff: [256]u8 = undefined;
        var fba = FixedBufferAllocator.init(&buff);
        const allocator = fba.allocator();
        index_map.ensureTotalCapacity(allocator, 32) catch unreachable;

        const queue_indices_as_slice: []const u8 = @ptrCast(&queue_indices);
        for (queue_indices_as_slice) |index| {
            const maybe_occur_count = index_map.get(index);
            if (maybe_occur_count) |occur_count| {
                index_map.putAssumeCapacity(index, occur_count + 1);
            } else {
                index_map.putAssumeCapacity(index, 1);
            }
        }

        var iter = index_map.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* > 0) {
                out_unique_indices.append(entry.key_ptr.*);
            }
        }
    }

    fn query_swapchain_support(device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR, out_support_info: *SwapchainSupportInfo) void {
        vk_check(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR.?(device, surface, &out_support_info.capabilities));

        var format_count: u32 = undefined;
        var present_mode_count: u32 = undefined;
        vk_check(c.vkGetPhysicalDeviceSurfaceFormatsKHR.?(device, surface, &format_count, null));
        vk_check(c.vkGetPhysicalDeviceSurfacePresentModesKHR.?(device, surface, &present_mode_count, null));

        util.assert_msg(out_support_info.formats.capacity >= format_count, "Should not exceed max capacity", .{});
        util.assert_msg(out_support_info.present_modes.capacity >= present_mode_count, "Should not exceed max capacity", .{});

        if (format_count > 0) {
            out_support_info.formats.resize(format_count);
            vk_check(c.vkGetPhysicalDeviceSurfaceFormatsKHR.?(device, surface, &format_count, out_support_info.formats.data()));
        }

        if (present_mode_count > 0) {
            out_support_info.present_modes.resize(present_mode_count);
            vk_check(c.vkGetPhysicalDeviceSurfacePresentModesKHR.?(device, surface, &present_mode_count, out_support_info.present_modes.data()));
        }
    }

    fn find_queue_families(physical_device: c.VkPhysicalDevice, out_indices: *QueueFamilyIndices, surface: c.VkSurfaceKHR) bool {
        var queue_family_count: u32 = undefined;
        c.vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &queue_family_count, null);

        const MAX_QUEUE_FAMILIES = 16;

        util.assert_msg(MAX_QUEUE_FAMILIES >= queue_family_count, "Should not exceed max capacity", .{});

        var queue_families: FixedArray(c.VkQueueFamilyProperties, MAX_QUEUE_FAMILIES) = .init_len(queue_family_count);
        c.vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &queue_family_count, queue_families.data());

        const queue_family_slice = queue_families.slice_full();

        if (pick_queue(queue_family_slice, c.VK_QUEUE_GRAPHICS_BIT, 0, c.VK_QUEUE_TRANSFER_BIT)) |queue_index| {
            out_indices.graphics = queue_index;
        } else {
            logger.log_error(.FATAL, "appropriate queue for Graphics was not found on selected vulkan device", .{});
            return false;
        }

        if (pick_queue(queue_family_slice, c.VK_QUEUE_TRANSFER_BIT, 0, c.VK_QUEUE_GRAPHICS_BIT)) |queue_index| {
            out_indices.transfer = queue_index;
        } else {
            logger.log_error(.FATAL, "appropriate queue for Transfer was not found on selected vulkan device", .{});
            return false;
        }

        if (pick_queue(queue_family_slice, c.VK_QUEUE_COMPUTE_BIT, c.VK_QUEUE_GRAPHICS_BIT, c.VK_QUEUE_TRANSFER_BIT)) |queue_index| {
            out_indices.compute = queue_index;
        } else {
            logger.log_error(.FATAL, "appropriate queue for Computing was not found on selected vulkan device", .{});
            return false;
        }

        if (pick_presentation_queue(queue_family_slice, physical_device, surface, out_indices.graphics)) |queue_index| {
            out_indices.present = queue_index;
        } else {
            logger.log_error(.FATAL, "appropriate queue for Presentation was not found on selected vulkan device", .{});
            return false;
        }

        return true;
    }

    const PickQueueReturn = struct {
        index: ?u8,
        similar_queue_index_match: bool,
    };

    fn pick_queue(
        queue_families: []c.VkQueueFamilyProperties,
        required_flag: c.VkQueueFlagBits,
        optional_flags: c.VkQueueFlagBits,
        not_needed_flags: c.VkQueueFlagBits,
    ) ?u8 {
        var max_score: u32 = 0;
        var result_index: ?u8 = null;
        var curr_score: u32 = undefined;

        for (queue_families, 0..queue_families.len) |family, i| {
            if ((family.queueFlags & required_flag) != required_flag) {
                continue;
            }

            curr_score = 1;

            if (optional_flags > 0 and (family.queueFlags & optional_flags) == optional_flags) {
                curr_score += 1;
            }

            if ((family.queueFlags & not_needed_flags) == 0) {
                curr_score += 1;
            }

            if (curr_score > max_score) {
                result_index = @intCast(i);
                max_score = curr_score;
            }
        }

        return result_index;
    }

    fn pick_presentation_queue(
        queue_families: []c.VkQueueFamilyProperties,
        phys_device: c.VkPhysicalDevice,
        surface: c.VkSurfaceKHR,
        already_picked_graphics_queue: u8,
    ) ?u8 {
        var max_score: u32 = 0;
        var result_index: ?u8 = null;
        var curr_score: u32 = undefined;

        for (queue_families, 0..queue_families.len) |family, i| {
            var supports_presentation: u32 = undefined;
            vk_check(c.vkGetPhysicalDeviceSurfaceSupportKHR.?(phys_device, @intCast(i), surface, &supports_presentation));

            if (supports_presentation == 0) {
                continue;
            }

            curr_score = 1;

            if (family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT > 0) {
                curr_score += 1;
            }

            if (i == already_picked_graphics_queue) {
                curr_score += 1;
            }

            if (curr_score > max_score) {
                result_index = @intCast(i);
                max_score = curr_score;
            }
        }

        return result_index;
    }
};
