const std = @import("std");
const builtin = @import("builtin");
const log_mod = @import("logger.zig");

pub fn FixedArray(comptime T: type, CAPACITY: u32) type {
    return struct {
        const Self = @This();
        const is_destructible: bool = std.meta.hasMethod(T, "deinit");

        comptime capacity: u32 = CAPACITY,
        len: u32 = 0,
        items: [CAPACITY]T = undefined,

        pub const empty = Self{};

        pub fn init_len(input_len: u32) Self {
            std.debug.assert(input_len <= CAPACITY);
            return Self{ .len = input_len };
        }

        pub fn init_len_to_cap() Self {
            return Self{ .len = CAPACITY };
        }

        pub fn init_from_slice(input_slice: []const T) Self {
            var new_arr = empty;
            const append_items = blk: {
                if (input_slice.len > CAPACITY) {
                    break :blk input_slice[0..CAPACITY];
                } else {
                    break :blk input_slice;
                }
            };

            new_arr.append_many(append_items);
            return new_arr;
        }

        pub fn append(self: *Self, item: T) void {
            if (self.len == self.capacity) {
                return;
            }

            self.items[self.len] = item;
            self.len += 1;
        }

        pub fn append_many(self: *Self, items: []const T) void {
            std.debug.assert(items.len <= (self.capacity - self.len));

            for (items) |item| {
                self.append(item);
            }
        }

        pub inline fn pop(self: *Self) void {
            std.debug.assert(self.len >= 1);

            if (is_destructible) {
                self.ptr_last().deinit();
            }
            self.len -= 1;
        }

        pub inline fn pop_many(self: *Self, pop_len: u32) void {
            std.debug.assert(pop_len <= self.len);

            if (is_destructible) {
                for (self.slice_len(self.len - pop_len, pop_len)) |*item| {
                    item.deinit();
                }
            }
            self.len -= pop_len;
        }

        pub fn remove(self: *Self, index: u32) void {
            std.debug.assert(index < self.len);

            if (index == self.len - 1) {
                self.pop();
                return;
            }

            if (is_destructible) {
                const item = self.ptr_offset(index);
                item.deinit();
            }

            @memmove(self.ptr_offset(index), self.slice(index + 1, self.len));
            @memset(self.slice_len(self.len - 1, 1), std.mem.zeroes(T));
            self.len -= 1;
        }

        pub fn remove_unordered(self: *Self, index: u32) void {
            std.debug.assert(index < self.len);
            if (index == self.len - 1) {
                self.pop();
                return;
            }

            const remove_item = self.ptr_offset(index);
            if (is_destructible) {
                remove_item.deinit();
            }

            std.mem.swap(T, remove_item, self.ptr_last());
            self.len -= 1;
        }

        pub inline fn slice(self: *Self, start: u32, end: u32) []T {
            std.debug.assert(start >= 0 and end <= self.len);
            return self.items[start..end];
        }

        pub inline fn slice_len(self: *Self, start: u32, len: u32) []T {
            std.debug.assert(start >= 0 and len <= (self.len - start));
            return self.items[start..(start + len)];
        }

        pub inline fn slice_as_const(self: *const Self, start: u32, end: u32) []const T {
            std.debug.assert(start >= 0 and end <= self.len);
            return self.items[start..end];
        }

        pub inline fn slice_len_as_const(self: *const Self, start: u32, len: u32) []const T {
            std.debug.assert(start >= 0 and len <= (self.len - start));
            return self.items[start..(start + len)];
        }

        pub inline fn slice_full(self: *Self) []T {
            return self.items[0..self.len];
        }

        pub inline fn slice_full_as_const(self: *const Self) []const T {
            return self.items[0..self.len];
        }

        pub inline fn clear(self: *Self) void {
            self.len = 0;
        }

        pub inline fn resize(self: *Self, input_len: u32) void {
            std.debug.assert(input_len <= self.capacity);
            self.len = input_len;
        }

        pub inline fn resize_to_cap(self: *Self) void {
            self.len = self.capacity;
        }

        pub inline fn ptr(self: *Self) *T {
            return @ptrCast(&self.items);
        }

        pub inline fn ptr_offset(self: *Self, offset: u32) *T {
            std.debug.assert(offset < self.len);
            return @ptrCast(&self.items[offset]);
        }

        pub inline fn ptr_as_const(self: *const Self) *const T {
            return @ptrCast(&self.items);
        }

        pub inline fn ptr_offset_as_const(self: *const Self, offset: u32) *const T {
            std.debug.assert(offset < self.len);
            return @ptrCast(&self.items[offset]);
        }

        pub inline fn ptr_last(self: *Self) *T {
            return &self.items[self.len - 1];
        }

        pub inline fn ptr_last_as_const(self: *const Self) *const T {
            return &self.items[self.len - 1];
        }

        pub inline fn c_ptr(self: *Self) [*c]T {
            return @ptrCast(&self.items);
        }

        pub inline fn c_ptr_as_const(self: *const Self) [*c]const T {
            return @ptrCast(&self.items);
        }

        pub fn print(self: *const Self) void {
            log_mod.log_debug(.DEBUG, "{any}", .{self.items});
        }
    };
}
