const std = @import("std");
const c = @import("c.zig").c;
const device_mod = @import("vk_device.zig");
const VulkanDevice = device_mod.VulkanDevice;
const FixedArray = @import("fixed_array.zig").FixedArray;
const VulkanImage = @import("vk_image.zig").VulkanImage;
const vk_check = @import("vk_shared.zig").vk_check;
const util = @import("util.zig");
const logger = @import("logger.zig");

pub const VulkanSwapchain = struct {
    const IMAGES_IN_FLIGHT = 2;
    const IMAGES_PER_FRAME = 1 + IMAGES_IN_FLIGHT;
    const MAX_IMAGE_COUNT = 8;
    const Self = @This();

    handle: c.VkSwapchainKHR,
    images: FixedArray(c.VkImage, MAX_IMAGE_COUNT),
    image_views: FixedArray(c.VkImageView, MAX_IMAGE_COUNT),
    image_format: c.VkSurfaceFormatKHR,
    depth_image: VulkanImage,

    pub fn init(
        device: *const VulkanDevice,
        surface: c.VkSurfaceKHR,
        maybe_swapchain_extent: ?c.VkExtent2D,
        out_swapchain: *VulkanSwapchain,
    ) bool {
        var found = false;
        const support_info = &device.swapchain_support_info;

        for (support_info.formats.slice_full_as_const()) |format| {
            if (format.format == c.VK_FORMAT_B8G8R8A8_SRGB and format.colorSpace == c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                found = true;
                out_swapchain.image_format = format;
                break;
            }
        }

        if (!found) {
            out_swapchain.image_format = support_info.formats.items[0];
        }

        var present_mode: c.VkPresentModeKHR = @intCast(c.VK_PRESENT_MODE_FIFO_KHR);
        for (device.swapchain_support_info.present_modes.slice_full_as_const()) |curr_present_mode| {
            if (curr_present_mode == @as(c_uint, @intCast(c.VK_PRESENT_MODE_MAILBOX_KHR))) {
                present_mode = curr_present_mode;
                break;
            }
        }

        var extent: c.VkExtent2D = undefined;
        if (maybe_swapchain_extent) |swapchain_extent| {
            extent = swapchain_extent;
            const min_extent = support_info.capabilities.minImageExtent;
            const max_extent = support_info.capabilities.maxImageExtent;
            extent.width = std.math.clamp(extent.width, min_extent.width, max_extent.width);
            extent.height = std.math.clamp(extent.height, min_extent.height, max_extent.height);
        } else if (support_info.capabilities.currentExtent.width != 0xffffffff) {
            extent = support_info.capabilities.currentExtent;
        }

        var create_image_count = support_info.capabilities.minImageCount + 1;
        if (support_info.capabilities.maxImageCount > 0 and create_image_count > support_info.capabilities.maxImageCount) {
            create_image_count = support_info.capabilities.maxImageCount;
        }

        var create_info: c.VkSwapchainCreateInfoKHR = .{
            .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = create_image_count,
            .imageFormat = out_swapchain.image_format.format,
            .imageColorSpace = out_swapchain.image_format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = support_info.capabilities.currentTransform,
            .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode,
            .oldSwapchain = null,
        };

        if (device.queue_family_indices.graphics != device.queue_family_indices.present) {
            const queue_indices = [2]u32{ @intCast(device.queue_family_indices.graphics), @intCast(device.queue_family_indices.present) };
            create_info.imageSharingMode = c.VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = queue_indices.len;
            create_info.pQueueFamilyIndices = @ptrCast(&queue_indices);
        } else {
            create_info.imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
            create_info.queueFamilyIndexCount = 0;
            create_info.pQueueFamilyIndices = null;
        }

        vk_check(c.vkCreateSwapchainKHR.?(device.logical_device, &create_info, null, &out_swapchain.handle));

        var actual_image_count: u32 = undefined;
        vk_check(c.vkGetSwapchainImagesKHR.?(device.logical_device, out_swapchain.handle, &actual_image_count, null));

        std.debug.assert(actual_image_count <= MAX_IMAGE_COUNT);

        if (out_swapchain.images.len != actual_image_count) {
            out_swapchain.images.resize(actual_image_count);
        }
        if (out_swapchain.image_views.len != actual_image_count) {
            out_swapchain.image_views.resize(actual_image_count);
        }

        vk_check(c.vkGetSwapchainImagesKHR.?(device.logical_device, out_swapchain.handle, &actual_image_count, out_swapchain.images.data()));

        for (0..out_swapchain.image_views.len) |i| {
            out_swapchain.image_views.items[i] = VulkanImage.create_view_from_raw_image(
                out_swapchain.images.items[i],
                device.logical_device,
                c.VK_IMAGE_TYPE_2D,
                out_swapchain.image_format.format,
                c.VK_IMAGE_ASPECT_COLOR_BIT,
            );
        }

        if (!VulkanImage.init(
            device,
            &out_swapchain.depth_image,
            c.VK_IMAGE_TYPE_2D,
            extent.width,
            extent.height,
            device.depth_format,
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            c.VK_IMAGE_ASPECT_DEPTH_BIT,
            c.VK_IMAGE_LAYOUT_UNDEFINED,
            c.VK_SAMPLE_COUNT_1_BIT,
            true,
        )) {
            logger.log_error(.FATAL, "Failed to create swapchain depth attachment", .{});
            return false;
        }

        logger.log_debug(.INFO, "Swapchain is successfully created!", .{});
        return true;
    }

    pub fn deinit(self: *Self, device: c.VkDevice) void {
        self.depth_image.deinit(device);

        for (self.image_views.slice_full()) |view| {
            c.vkDestroyImageView.?(device, view, null);
        }

        self.images.clear();
        self.image_views.clear();

        if (self.handle != null) {
            c.vkDestroySwapchainKHR.?(device, self.handle, null);
            self.handle = null;
        }
    }
};
