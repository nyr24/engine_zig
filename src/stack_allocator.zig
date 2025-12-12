const std = @import("std");
const clib = std.c;
const Allocator = std.mem.Allocator;
const Error = Allocator.Error;
const Alignment = std.mem.Alignment;
const util = @import("util.zig");

const Config = struct {
    grow_factor: f32 = 2.0,
};

pub fn StackAllocator(comptime config: Config) type {
    return struct {
        const Self = @This();
        const DEFAULT_ALIGNMENT: usize = @alignOf(*anyopaque);
        const DEFAULT_CAPACITY: usize = 4096;

        const Header = struct {
            prev_alloc_len: u32,
            padding: u16,
        };

        const LastAllocReturn = struct {
            len: u32,
            is_last: bool,
        };

        pub const vtable: Allocator.VTable = .{
            .alloc = alloc,
            .free = free,
            .remap = remap,
            .resize = resize,
        };

        buffer: ?[*]u8 = null,
        prev_len: usize = 0,
        len: usize = 0,
        cap: usize = DEFAULT_CAPACITY,

        pub fn init_capacity(init_cap: usize) !Self {
            var self = Self{ .cap = init_cap };
            const unaligned_ptr: [*]u8 = @ptrCast(clib.malloc(self.cap + DEFAULT_ALIGNMENT - 1) orelse return Error.OutOfMemory);
            const unaligned_addr = @intFromPtr(unaligned_ptr);
            const aligned_addr = std.mem.alignForward(usize, unaligned_addr, DEFAULT_ALIGNMENT);
            self.buffer = unaligned_ptr;
            self.len += aligned_addr - unaligned_addr;
            self.prev_len = self.len;
            return self;
        }

        pub fn alloc(context: *anyopaque, byte_size: usize, alignment: Alignment, ret_address: usize) ?[*]u8 {
            _ = ret_address;

            var self: *Self = @ptrCast(@alignCast(context));
            std.debug.assert(self.buffer != null);
            const buffer = self.buffer.?;
            const curr_addr = @intFromPtr(buffer) + self.len;
            const curr_addr_with_header = curr_addr + @sizeOf(Header);
            const aligned_addr = std.mem.alignForward(usize, curr_addr_with_header, alignment.toByteUnits());
            const padding = aligned_addr - curr_addr;

            if ((byte_size + padding) > self.remain_cap()) {
                self.resize_inner(self.len + byte_size + padding, alignment) catch return null;
            }

            const header: *Header = @ptrFromInt(aligned_addr - @sizeOf(Header));
            header.prev_alloc_len = @intCast(self.len - self.prev_len);
            header.padding = @intCast(padding);
            const res_offset = self.len + padding;
            const res = buffer + res_offset;
            self.prev_len = self.len;
            self.len += padding + byte_size;
            return res;
        }

        pub fn free(context: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
            _ = ret_addr;
            _ = alignment;

            var self: *Self = @ptrCast(@alignCast(context));
            std.debug.assert(self.buffer != null);

            const last_alloc_info = self.is_last_alloc(memory);

            if (!last_alloc_info.is_last) {
                return;
            }

            self.len = self.prev_len;
            self.prev_len -= last_alloc_info.len;
        }

        pub fn resize(context: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
            _ = ret_addr;
            _ = alignment;

            var self: *Self = @ptrCast(@alignCast(context));
            std.debug.assert(new_len > 0);

            const len_diff = new_len - memory.len;
            const grow_needed = len_diff > 0;
            const last_alloc_info = self.is_last_alloc(memory);

            if (!last_alloc_info.is_last) {
                if (!grow_needed) {
                    return true;
                } else {
                    return false;
                }
            }

            if (len_diff > self.remain_cap()) {
                return false;
            }

            self.len += len_diff;
            return true;
        }

        fn remap(context: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
            _ = ret_addr;

            var self: *Self = @ptrCast(@alignCast(context));
            std.debug.assert(new_len > 0);

            const len_diff = new_len - memory.len;
            const grow_needed = len_diff > 0;
            const last_alloc_info = self.is_last_alloc(memory);

            if (!last_alloc_info.is_last) {
                if (!grow_needed) {
                    return memory.ptr;
                }
                const new_alloc = alloc(context, new_len, alignment, 0) orelse return null;
                @memcpy(new_alloc[0..new_len], memory);
                return new_alloc;
            }

            if (len_diff > self.remain_cap()) {
                self.resize_inner(self.len + len_diff, alignment) catch return null;
            }

            self.len += len_diff;
            return memory.ptr;
        }

        pub fn allocator(self: *Self) Allocator {
            return .{
                .ptr = @ptrCast(@alignCast(self)),
                .vtable = &vtable,
            };
        }

        fn is_last_alloc(self: *Self, memory: []u8) LastAllocReturn {
            std.debug.assert(self.buffer != null);
            const buffer = self.buffer.?;
            if (!util.is_address_in_range(memory.ptr, buffer, self.cap)) {
                return .{ .is_last = false, .len = 0 };
            }
            const memory_ptr: [*]u8 = memory.ptr;
            const header: *Header = @ptrCast(@alignCast(memory_ptr - @sizeOf(Header)));
            const maybe_last_alloc_offset = util.ptr_as_offset(memory_ptr - header.padding, buffer);
            return .{
                .len = header.prev_alloc_len,
                .is_last = maybe_last_alloc_offset == self.prev_len,
            };
        }

        fn resize_inner(self: *Self, hint_cap: usize, alignment: Alignment) !void {
            _ = alignment;
            std.debug.assert(self.buffer != null);
            const grow_cap_int: usize = @intFromFloat(@as(f64, @floatFromInt(self.cap)) * @as(f64, config.grow_factor));
            self.cap = @max(grow_cap_int, hint_cap);
            self.buffer = @ptrCast(clib.realloc(self.buffer, self.cap) orelse return Error.OutOfMemory);
        }

        fn remain_cap(self: *Self) usize {
            return self.cap - self.len;
        }

        pub fn deinit(self: *Self) void {
            if (self.buffer != null) {
                clib.free(self.buffer);
                self.buffer = null;
            }
        }
    };
}
