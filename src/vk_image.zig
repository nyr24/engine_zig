const std = @import("std");
const c = @import("c.zig").c;
const VulkanDevice = @import("vk_device.zig").VulkanDevice;
const FixedArray = @import("fixed_array.zig").FixedArray;
const shared_mod = @import("vk_shared.zig");
const vk_check = shared_mod.vk_check;
const util = @import("util.zig");
const logger = @import("logger.zig");

pub const VulkanImage = struct {
    handle: c.VkImage,
    view: c.VkImageView,
    memory: c.VkDeviceMemory,
    width: u32,
    height: u32,
    type: c.VkImageType = c.VK_IMAGE_TYPE_2D,
    format: c.VkFormat,

    const Self = @This();

    pub fn init(
        device: *const VulkanDevice,
        out_image: *Self,
        image_type: c.VkImageType,
        width: u32,
        height: u32,
        format: c.VkFormat,
        tiling: c.VkImageTiling,
        usage: c.VkImageUsageFlags,
        memory_flags: c.VkMemoryPropertyFlags,
        aspect_flags: c.VkImageAspectFlags,
        init_layout: c.VkImageLayout,
        sample_count: c.VkSampleCountFlagBits,
        should_create_view: bool,
    ) bool {
        const create_info: c.VkImageCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = null,
            .imageType = image_type,
            .format = format,
            .extent = .{
                .width = width,
                .height = height,
                .depth = @as(u32, 1),
            },
            .mipLevels = 4,
            .arrayLayers = 1,
            .samples = sample_count,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = init_layout,
        };

        vk_check(c.vkCreateImage.?(device.logical_device, &create_info, null, &out_image.handle));

        out_image.width = width;
        out_image.height = height;
        out_image.type = image_type;
        out_image.format = format;

        var memory_requirements: c.VkMemoryRequirements = undefined;
        c.vkGetImageMemoryRequirements.?(device.logical_device, out_image.handle, &memory_requirements);

        const memory_type_index = device.find_memory_index(memory_requirements.memoryTypeBits, memory_flags) orelse {
            logger.log_error(.FATAL, "Required memory type not found. Image not valid", .{});
            return false;
        };

        const alloc_info: c.VkMemoryAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = memory_requirements.size,
            .memoryTypeIndex = memory_type_index,
        };

        vk_check(c.vkAllocateMemory.?(device.logical_device, &alloc_info, null, &out_image.memory));
        vk_check(c.vkBindImageMemory.?(device.logical_device, out_image.handle, out_image.memory, 0));

        if (should_create_view) {
            out_image.create_view(device.logical_device, aspect_flags);
        }

        return true;
    }

    pub fn create_view(self: *VulkanImage, device: c.VkDevice, aspect_flags: c.VkImageAspectFlags) void {
        self.view = create_view_from_raw_image(self.handle, device, self.type, self.format, aspect_flags);
    }

    pub fn create_view_from_raw_image(
        image: c.VkImage,
        device: c.VkDevice,
        image_type: c.VkImageType,
        format: c.VkFormat,
        aspect_flags: c.VkImageAspectFlags,
    ) c.VkImageView {
        var view: c.VkImageView = undefined;

        const create_info: c.VkImageViewCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = null,
            .image = image,
            .viewType = image_type,
            .format = format,
            .subresourceRange = .{
                .aspectMask = aspect_flags,
                .baseMipLevel = 0,
                .levelCount = c.VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        vk_check(c.vkCreateImageView.?(device, &create_info, null, &view));
        return view;
    }

    pub fn deinit(self: *Self, device: c.VkDevice) void {
        if (self.handle != null) {
            c.vkDestroyImageView.?(device, self.view, null);
            c.vkFreeMemory.?(device, self.memory, null);
            c.vkDestroyImage.?(device, self.handle, null);
            self.view = null;
            self.memory = null;
            self.handle = null;
        }
    }

    // TODO: after cmd_buffers
    // pub fn transition_layout_of_raw_image(
    //     image: c.VkImage,
    // ) void {}
};
