const std = @import("std");
const util = @import("util.zig");
const logger = @import("logger.zig");

// NOTE: All vectors, 2, 3, 4 merged into 1 type, because it's easy to share operations between them
// TODO: test between Self and *Self
pub fn Vec(comptime LEN: u32, comptime T: type) type {
    return struct {
        v: @Vector(LEN, T) = undefined,

        const Self = @This();
        pub const len = LEN;
        pub const zero_vec: Self = .{ .v = @splat(0) };
        pub const unit_vec: Self = .{ .v = @splat(1) };

        pub fn init_from_slice(data: *const [LEN]f32) Self {
            var res: Self = .{};
            for (data, 0..LEN) |val, i| {
                res.v[i] = val;
            }
            return res;
        }

        pub fn init_from_vec(input_v: @Vector(LEN, f32)) Self {
            return .{ .v = input_v };
        }

        pub fn init_dupl(val: f32) Self {
            return Self{ .v = @splat(val) };
        }

        pub fn init_zero() Self {
            return zero_vec;
        }

        pub fn init_one() Self {
            return unit_vec;
        }

        pub fn set_from_slice(self: *Self, input: *const [LEN]f32) void {
            for (0..LEN) |i| {
                self.v[i] = input[i];
            }
        }

        pub fn set_from_vec(self: *Self, comptime INPUT_LEN: u32, input_vec: Vec(INPUT_LEN, f32)) void {
            const min_len = comptime @min(LEN, INPUT_LEN);
            for (0..min_len) |i| {
                self.v[i] = input_vec.v[i];
            }
        }

        pub inline fn x(self: Self) f32 {
            return self.v[0];
        }

        pub inline fn y(self: Self) f32 {
            return self.v[1];
        }

        pub inline fn z(self: Self) f32 {
            return self.v[2];
        }

        pub fn cross(lhs: Self, rhs: Self) Self {
            if (LEN == 3) {
                const lx, const ly, const lz = lhs.v;
                const rx, const ry, const rz = rhs.v;
                return .{ ly * rz - lz * ry, lx * rz - lz * rx, lx * ry - ly * rx };
            }
            unreachable;
        }

        pub fn dot(lhs: Self, rhs: Self) f32 {
            return @reduce(.Add, lhs.v * rhs.v);
        }

        pub fn mul(lhs: Self, rhs: Self) Self {
            return Self.init_from_vec(lhs.v * rhs.v);
        }

        pub fn mul_inplace(lhs: *Self, rhs: Self) void {
            lhs.v *= rhs.v;
        }

        pub fn reciprocal(self: Self) Self {
            var res: Vec3 = .{};
            for (0..LEN) |i| {
                res.v[i] = 1 / self.v[i];
            }
            return res;
        }

        pub fn reciprocal_inplace(self: *Self) void {
            for (0..LEN) |i| {
                self.v[i] = 1 / self.v[i];
            }
        }

        pub fn normalized(self: Self) Self {
            var res: Self = self;
            res.negate_inplace();
            return res;
        }

        pub fn normalize_inplace(self: *Self) void {
            const mag = self.magnitude();
            self.v *= 1 / mag;
        }

        pub inline fn magnitude(self: Self) f32 {
            return @sqrt(self.magnitude_squared());
        }

        pub inline fn magnitude_squared(self: Self) f32 {
            return self.dot(self);
        }

        pub fn distance(lhs: Self, rhs: Self) Self {
            return rhs.v - lhs.v;
        }

        pub fn negated(self: Self) Self {
            var res: Self = self;
            res.negate_inplace();
            return res;
        }

        pub inline fn negate_inplace(self: *Self) void {
            self.v *= @as(@Vector(LEN, f32), @splat(-1.0));
        }

        pub fn cmp(lhs: Self, rhs: Self) bool {
            const cmp_res = lhs.v == rhs.v;
            for (0..LEN) |i| {
                if (!cmp_res[i]) {
                    return false;
                }
            }
            return true;
        }

        pub fn print(self: Self) void {
            logger.log_debug(.INFO, "{}", .{self.v});
        }
    };
}

pub fn vec4_to_vec3(v: Vec4) Vec3 {
    return Vec3.init_from_slice(&.{ v.v[0], v.v[1], v.v[2] });
}

pub fn vec4_to_vec2(v: Vec4) Vec2 {
    return Vec2.init_from_slice(&.{ v.v[0], v.v[1] });
}

pub fn vec3_to_vec4(v: Vec3) Vec4 {
    return Vec4.init_from_slice(&.{ v.v[0], v.v[1], v.v[2], 1.0 });
}

pub fn vec3_to_vec2(v: Vec3) Vec2 {
    return Vec2.init_from_slice(&.{ v.v[0], v.v[1] });
}

pub const Vec2 = Vec(2, f32);
pub const Vec3 = Vec(3, f32);
pub const Vec4 = Vec(4, f32);

// Mat2x2 -----------------------------------------------------------------------------------
pub const Matrix2x2 = struct {
    const ROWS = 2;
    const COLS = 2;
    const ITEMS = ROWS * COLS;
    const Self = @This();
    const zero = std.mem.zeroes(Self);

    m: [ITEMS]f32,

    pub fn init_from_slice(init_slice: *const [ITEMS]f32) Self {
        var res: Self = undefined;
        for (0..ROWS) |y| {
            for (0..COLS) |x| {
                res.set_at(init_slice[y * COLS + x], y, x);
            }
        }
        return res;
    }

    pub fn init_from_vec(input_vec: *const @Vector(ITEMS, f32)) Self {
        var res: Self = .zero;
        for (0..ROWS) |y| {
            for (0..COLS) |x| {
                res.set_at(input_vec[y * COLS + x], y, x);
            }
        }
        return res;
    }

    pub fn init_identity() Self {
        var res: Self = undefined;
        res.identity_inplace();
        return res;
    }

    pub fn identity_inplace(self: *Self) void {
        self.* = .zero;
        self.set_at(1.0, 0, 0);
        self.set_at(1.0, 1, 1);
    }

    pub fn determinant(self: Self) f32 {
        return (self.m[0] * self.m[3]) - (self.m[2] * self.m[1]);
    }

    pub inline fn set_row(self: *Self, input_row: Vec2, index: usize) void {
        self.m[index] = input_row;
    }

    pub inline fn increment_row(self: *Self, input_row: Vec2, index: usize) void {
        self.m[index].v += input_row.v;
    }

    pub inline fn set_col(self: *Self, input_col: Vec2, index: usize) void {
        for (0..ROWS) |i| {
            self.set_at(input_col.v[i], i, index);
        }
    }

    pub inline fn increment_col(self: *Self, input_col: Vec2, index: usize) void {
        for (0..ROWS) |i| {
            self.increment_at(input_col.v[i], i, index);
        }
    }

    pub inline fn get_row(self: *const Self, index: usize) Vec2 {
        return self.m[index];
    }

    pub fn get_col(self: *const Self, index: usize) Vec2 {
        var col: Vec2 = .{};
        for (0..ROWS) |i| {
            col.v[i] = self.get_at(i, index);
        }
        return col;
    }

    pub inline fn set_at(self: *Self, val: f32, row: usize, col: usize) void {
        self.m[row].v[col] = val;
    }

    pub inline fn increment_at(self: *Self, val: f32, row: usize, col: usize) void {
        self.m[row].v[col] += val;
    }

    pub inline fn get_at(self: *const Self, row: usize, col: usize) f32 {
        return self.m[row].v[col];
    }

    pub inline fn get_at_ptr(self: *Self, row: usize, col: usize) *f32 {
        return &self.m[row].v[col];
    }

    pub inline fn get_at_ptr_as_const(self: *const Self, row: usize, col: usize) *const f32 {
        return &self.m[row].v[col];
    }

    pub inline fn as_single_vec(self: *Self) *@Vector(ITEMS, f32) {
        return @ptrCast(@alignCast(&self.m));
    }

    pub inline fn as_single_vec_as_const(self: *const Self) *const @Vector(ITEMS, f32) {
        return @ptrCast(@alignCast(&self.m));
    }

    pub fn print(self: Self) void {
        for (0..ROWS) |y| {
            self.get_row(y).print();
        }
    }
};

// Mat3x3 -----------------------------------------------------------------------------------
// Laplace expansion theorem (inverse): https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
pub const Matrix3x3 = struct {
    const ROWS = 3;
    const COLS = 3;
    const ITEMS = ROWS * COLS;
    const Self = @This();
    const zero = std.mem.zeroes(Self);

    m: [ROWS]Vec3,

    pub fn init_identity() Self {
        var res: Self = undefined;
        res.identity_inplace();
        return res;
    }

    pub fn init_from_slice(init_slice: *const [ITEMS]f32) Self {
        var res: Self = undefined;
        for (0..ROWS) |y| {
            for (0..COLS) |x| {
                res.m[y].v[x] = init_slice[y * COLS + x];
            }
        }
        return res;
    }

    pub fn init_from_mat4x4(mat4: *const Matrix4x4) Self {
        var res: Self = undefined;
        for (0..ROWS) |y| {
            res.set_row(mat4.get_row_as_vec3(y), y);
        }
        return res;
    }

    pub fn init_from_mat2x2(mat2: *const Matrix2x2) Self {
        var res: Self = undefined;
        for (0..Matrix2x2.ROWS) |y| {
            for (0..Matrix2x2.COLS) |x| {
                res.set_at(mat2.get_at(y, x), y, x);
            }
        }
        res.set_col(.init_from_slice(&.{ 0, 0, 1 }), 2);
        res.set_row(.init_from_slice(&.{ 0, 0, 1 }), 2);
        return res;
    }

    pub fn identity_inplace(self: *Self) void {
        self.* = .zero;
        self.m[0].v[0] = 1.0;
        self.m[1].v[1] = 1.0;
        self.m[2].v[2] = 1.0;
    }

    pub fn mul(lhs: Self, rhs: Self) Self {
        var res: Self = .init_identity();

        for (0..ROWS) |y| {
            const lv = lhs.get_row(y);
            var res_row: Vec3 = undefined;
            for (0..COLS) |x| {
                const rv: Vec3 = rhs.get_col(x);
                res_row.v[x] = lv.dot(rv);
            }
            res.set_row(res_row, y);
        }

        return res;
    }

    pub fn mul_inplace(self: *Self, rhs: Self) void {
        for (0..ROWS) |y| {
            const lv = self.m[y];
            var res_row: Vec4 = undefined;
            for (0..COLS) |x| {
                const rv: Vec4 = rhs.get_col(x);
                res_row.v[x] = lv.dot(rv);
            }
            self.set_row(res_row, y);
        }
    }

    pub fn mul_vec3(lhs: Self, rhs: Vec3) Vec3 {
        var res: Vec3 = undefined;

        for (0..ROWS) |y| {
            const lhs_row = lhs.get_row(y);
            res.v[y] = lhs_row.dot(rhs);
        }

        return res;
    }

    pub fn inversed(self: Self) Self {
        var res = self;
        res.inverse_inplace();
        return res;
    }

    // NOTE: Laplace Theorem: https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    pub fn inverse_inplace(s: *Self) void {
        var adj: Matrix3x3 = undefined;

        adj.set_at(s.at(1, 1) * s.at(2, 2) - s.at(1, 2) * s.at(2, 1), 0, 0);
        adj.set_at(s.at(1, 0) * s.at(2, 2) - s.at(1, 2) * s.at(2, 0), 0, 1);
        adj.set_at(s.at(1, 0) * s.at(2, 1) - s.at(1, 1) * s.at(2, 0), 0, 2);
        adj.set_at(s.at(2, 2) * s.at(0, 1) - s.at(2, 1) * s.at(0, 2), 1, 0);
        adj.set_at(s.at(0, 0) * s.at(2, 2) - s.at(0, 2) * s.at(2, 0), 1, 1);
        adj.set_at(s.at(2, 1) * s.at(0, 0) - s.at(2, 0) * s.at(0, 1), 1, 2);
        adj.set_at(s.at(0, 1) * s.at(1, 2) - s.at(0, 2) * s.at(1, 1), 2, 0);
        adj.set_at(s.at(0, 0) * s.at(1, 2) - s.at(0, 2) * s.at(1, 0), 2, 1);
        adj.set_at(s.at(0, 0) * s.at(1, 1) - s.at(0, 1) * s.at(1, 0), 2, 2);

        // check if determinant is 0
        const det = (s.at(0, 0) * adj.at(0, 0)) + (s.at(0, 1) * adj.at(1, 0)) + (s.at(0, 2) * adj.at(2, 0));
        if (@abs(det) <= std.math.floatEps(f32)) {
            s.identity_inplace(); // cannot inverse, make it identity matrix
            return;
        }

        // divide by the determinant
        const inv_det_v: Vec3 = .init_dupl(1.0 / det);

        // multiply 'adj' matrix memberwise with inv_det_v and set to 's'
        s.set_row(adj.get_row_ptr(0).mul(inv_det_v), 0);
        s.set_row(adj.get_row_ptr(1).mul(inv_det_v), 1);
        s.set_row(adj.get_row_ptr(2).mul(inv_det_v), 2);
    }

    pub fn transposed(self: Self) Self {
        var res: Self = self;
        res.transpose_inplace();
        return res;
    }

    pub fn transpose_inplace(self: *Self) void {
        std.mem.swap(f32, self.get_at_ptr(0, 1), self.get_at_ptr(1, 0));
        std.mem.swap(f32, self.get_at_ptr(0, 2), self.get_at_ptr(2, 0));
        std.mem.swap(f32, self.get_at_ptr(1, 2), self.get_at_ptr(2, 1));
    }

    pub inline fn set_row(self: *Self, input_row: Vec3, index: usize) void {
        self.m[index] = input_row;
    }

    pub inline fn increment_row(self: *Self, input_row: Vec3, index: usize) void {
        self.m[index].v += input_row.v;
    }

    pub inline fn set_col(self: *Self, input_col: Vec3, index: usize) void {
        for (0..ROWS) |i| {
            self.set_at(input_col.v[i], i, index);
        }
    }

    pub inline fn increment_col(self: *Self, input_col: Vec3, index: usize) void {
        for (0..ROWS) |i| {
            self.increment_at(input_col.v[i], i, index);
        }
    }

    pub inline fn get_row(self: *const Self, index: usize) Vec3 {
        return self.m[index];
    }

    pub inline fn get_row_ptr(self: *Self, index: usize) *Vec3 {
        return &self.m[index];
    }

    pub inline fn get_row_ptr_as_const(self: *const Self, index: usize) *const Vec3 {
        return &self.m[index];
    }

    pub fn get_col(self: *const Self, index: usize) Vec3 {
        var col: Vec3 = .{};
        for (0..ROWS) |i| {
            col.v[i] = self.get_at(i, index);
        }
        return col;
    }

    pub fn get_translation_vec(self: *const Self) Vec3 {
        return self.get_col(3);
    }

    pub fn get_scaling_vec(self: *const Self) Vec3 {
        var scale_vec: Vec3 = undefined;
        scale_vec.v[0] = self.get_at(0, 0);
        scale_vec.v[1] = self.get_at(1, 1);
        scale_vec.v[2] = self.get_at(2, 2);
        return scale_vec;
    }

    pub fn set_scaling_vec(self: *Self, scale_vec: Vec3) void {
        self.set_at(scale_vec.v[0], 0, 0);
        self.set_at(scale_vec.v[1], 1, 1);
        self.set_at(scale_vec.v[2], 2, 2);
    }

    pub inline fn set_at(self: *Self, val: f32, row: usize, col: usize) void {
        self.m[row].v[col] = val;
    }

    pub inline fn increment_at(self: *Self, val: f32, row: usize, col: usize) void {
        self.m[row].v[col] += val;
    }

    pub inline fn get_at(self: *const Self, row: usize, col: usize) f32 {
        return self.m[row].v[col];
    }

    // shorter alias for get_at
    pub inline fn at(self: *const Self, row: usize, col: usize) f32 {
        return self.m[row].v[col];
    }

    pub inline fn get_at_ptr(self: *Self, row: usize, col: usize) *f32 {
        return &self.m[row].v[col];
    }

    pub inline fn get_at_ptr_as_const(self: *const Self, row: usize, col: usize) *const f32 {
        return &self.m[row].v[col];
    }

    pub inline fn as_single_vec(self: *Self) *@Vector(ITEMS, f32) {
        return @ptrCast(@alignCast(&self.m));
    }

    pub inline fn as_single_vec_as_const(self: *const Self) *const @Vector(ITEMS, f32) {
        return @ptrCast(@alignCast(&self.m));
    }

    pub inline fn as_single_slice(self: *Self) []f32 {
        return @ptrCast(@alignCast(&self.m));
    }

    pub fn print(self: Self) void {
        for (0..ROWS) |y| {
            self.get_row(y).print();
        }
    }
};

// Mat4x4 -----------------------------------------------------------------------------------

pub const Axis = enum { X, Y, Z };

// NOTE: Matrix-vector multiplication scheme will be: Mat4x4 * Vec4
// NOTE: coordinate systems in Vulkan https://www.kdab.com/projection-matrices-with-vulkan-part-1/

// TODO: mat * vec multiplications
// TEST: test different representations ([16]f32, [4]@Vector(4, f32), @Vector(16, f32))
pub const Matrix4x4 = struct {
    const ROWS = 4;
    const COLS = 4;
    const ITEMS = ROWS * COLS;
    const Self = @This();
    const zero = std.mem.zeroes(Self);

    m: [ROWS]Vec4,

    pub fn init_from_slice(input_slice: *const [ITEMS]f32) Self {
        var res: Self = .zero;
        for (0..ROWS) |y| {
            for (0..COLS) |x| {
                res.set_at(input_slice[y * COLS + x], y, x);
            }
        }
        return res;
    }

    pub fn init_from_mat3x3(mat3: *const Matrix3x3) Self {
        var res: Self = undefined;
        for (0..Matrix3x3.ROWS) |y| {
            for (0..Matrix3x3.COLS) |x| {
                res.set_at(mat3.get_at(y, x), y, x);
            }
        }
        res.set_col(.init_from_slice(&.{ 0, 0, 0, 1 }), 3);
        res.set_row(.init_from_slice(&.{ 0, 0, 0, 1 }), 3);
        return res;
    }

    pub fn init_from_mat2x2(mat2: *const Matrix2x2) Self {
        var res: Self = undefined;
        for (0..Matrix2x2.ROWS) |y| {
            for (0..Matrix2x2.COLS) |x| {
                res.set_at(mat2.get_at(y, x), y, x);
            }
        }
        res.set_col(.init_from_slice(&.{ 0, 0, 1, 0 }), 2);
        res.set_col(.init_from_slice(&.{ 0, 0, 0, 1 }), 3);
        res.set_row(.init_from_slice(&.{ 0, 0, 1, 0 }), 2);
        res.set_row(.init_from_slice(&.{ 0, 0, 0, 1 }), 3);
        return res;
    }

    pub fn init_from_rows(input_rows: *const [ROWS]Vec4) Self {
        var res: Self = .zero;
        for (0..ROWS) |y| {
            res.set_row(.init_from_vec(input_rows[y]), y);
        }
        return res;
    }

    pub fn init_from_row_slices(input_rows: *const [ROWS][COLS]f32) Self {
        var res: Self = .zero;
        for (0..ROWS) |y| {
            res.set_row(.init_from_slice(&input_rows[y]), y);
        }
        return res;
    }

    pub fn init_identity() Self {
        var res: Self = undefined;
        res.identity_inplace();
        return res;
    }

    pub fn identity_inplace(self: *Self) void {
        self.* = .zero;
        self.set_at(1.0, 0, 0);
        self.set_at(1.0, 1, 1);
        self.set_at(1.0, 2, 2);
        self.set_at(1.0, 3, 3);
    }

    pub fn is_identity(self: Self) bool {
        for (0..ROWS) |y| {
            for (0..COLS) |x| {
                const val = self.get_at(y, x);
                if (y == x and val != 1.0) {
                    return false;
                } else if (val != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    pub fn init_translation(translation_vec: Vec3) Self {
        var res: Self = .init_identity();
        res.set_col(vec3_to_vec4(translation_vec), 3);
        return res;
    }

    pub fn set_translate_inplace(self: *Self, translation_vec: Vec3) void {
        self.set_col_from_vec3(translation_vec, 3);
    }

    pub fn increment_translate_inplace(self: *Self, translation_vec: Vec3) void {
        self.increment_col(vec3_to_vec4(translation_vec), 3);
    }

    pub fn init_scaling(scaling_vec: Vec3) Self {
        var res: Self = .init_identity();
        for (0..3) |i| {
            res.set_at(scaling_vec.v[i], i, i);
        }
        return res;
    }

    pub fn set_scale_inplace(self: *Self, scaling_vec: Vec3) void {
        for (0..3) |i| {
            self.set_at(scaling_vec.v[i], i, i);
        }
    }

    pub fn increment_scale_inplace(self: *Self, scaling_vec: Vec3) void {
        for (0..3) |i| {
            self.increment_at(scaling_vec.v[i], i, i);
        }
    }

    pub fn init_rotation(rotate_deg: f32, axis: Axis) Self {
        var res: Self = .init_identity();
        const sin = @sin(std.math.rad_per_deg * rotate_deg);
        const cos = @cos(std.math.rad_per_deg * rotate_deg);

        switch (axis) {
            .X => {
                res.set_at(cos, 1, 1);
                res.set_at(-sin, 1, 2);
                res.set_at(sin, 2, 1);
                res.set_at(cos, 2, 2);
            },
            .Y => {
                res.set_at(cos, 0, 0);
                res.set_at(sin, 0, 2);
                res.set_at(-sin, 2, 0);
                res.set_at(cos, 2, 2);
            },
            .Z => {
                res.set_at(cos, 0, 0);
                res.set_at(-sin, 0, 1);
                res.set_at(sin, 1, 0);
                res.set_at(cos, 1, 1);
            },
        }

        return res;
    }

    pub fn rotate_inplace(self: *Self, rotate_deg: f32, axis: Axis) void {
        const sin = @sin(std.math.rad_per_deg * rotate_deg);
        const cos = @cos(std.math.rad_per_deg * rotate_deg);

        switch (axis) {
            .X => {
                self.increment_at(cos, 1, 1);
                self.increment_at(-sin, 1, 2);
                self.increment_at(sin, 2, 1);
                self.increment_at(cos, 2, 2);
            },
            .Y => {
                self.increment_at(cos, 0, 0);
                self.increment_at(sin, 0, 2);
                self.increment_at(-sin, 2, 0);
                self.increment_at(cos, 2, 2);
            },
            .Z => {
                self.increment_at(cos, 0, 0);
                self.increment_at(-sin, 0, 1);
                self.increment_at(sin, 1, 0);
                self.increment_at(cos, 1, 1);
            },
        }
    }

    pub fn mul(lhs: Self, rhs: Self) Self {
        var res: Self = .init_identity();

        for (0..ROWS) |y| {
            const lv = lhs.m[y];
            var res_row: Vec4 = undefined;
            for (0..COLS) |x| {
                const rv: Vec4 = rhs.get_col(x);
                res_row.v[x] = lv.dot(rv);
            }
            res.set_row(res_row, y);
        }

        return res;
    }

    pub fn mul_inplace(self: *Self, rhs: Self) void {
        for (0..ROWS) |y| {
            const lv = self.m[y];
            var res_row: Vec4 = undefined;
            for (0..COLS) |x| {
                const rv: Vec4 = rhs.get_col(x);
                res_row.v[x] = lv.dot(rv);
            }
            self.set_row(res_row, y);
        }
    }

    pub fn mul_vec4(lhs: Self, rhs: Vec4) Vec4 {
        var res: Vec4 = undefined;

        for (0..ROWS) |y| {
            const lhs_row = lhs.get_row(y);
            res.v[y] = lhs_row.dot(rhs);
        }

        return res;
    }

    pub fn mul_many(first: Self, inputs: []const Self) Self {
        var res: Self = .init_identity();

        for (0..ROWS) |y| {
            var row: Vec4 = first.get_row(y);
            var col: Vec4 = undefined;
            var between_res: Vec4 = undefined;

            for (inputs) |input_mat| {
                for (0..COLS) |x| {
                    col = input_mat.get_col(x);
                    between_res.v[x] = row.dot(col);
                }
                row = between_res;
            }

            res.set_row(row, y);
        }

        return res;
    }

    pub fn mul_many_inplace(self: *Self, inputs: []const Self) void {
        for (0..ROWS) |y| {
            var row: Vec4 = self.get_row(y);
            var col: Vec4 = undefined;
            var between_res: Vec4 = undefined;

            for (inputs) |input_mat| {
                for (0..COLS) |x| {
                    col = input_mat.get_col(x);
                    between_res.v[x] = row.dot(col);
                }
                row = between_res;
            }

            self.set_row(row, y);
        }
    }

    pub fn transposed(self: Self) Self {
        var res: Self = self;
        res.transpose_inplace();
        return res;
    }

    pub fn transpose_inplace(self: *Self) void {
        std.mem.swap(f32, self.get_at_ptr(0, 1), self.get_at_ptr(1, 0));
        std.mem.swap(f32, self.get_at_ptr(0, 2), self.get_at_ptr(2, 0));
        std.mem.swap(f32, self.get_at_ptr(0, 3), self.get_at_ptr(3, 0));
        std.mem.swap(f32, self.get_at_ptr(1, 2), self.get_at_ptr(2, 1));
        std.mem.swap(f32, self.get_at_ptr(1, 3), self.get_at_ptr(3, 1));
        std.mem.swap(f32, self.get_at_ptr(2, 3), self.get_at_ptr(3, 2));
    }

    pub fn inversed(self: Self) Self {
        var res = self;
        res.inverse_inplace();
        return res;
    }

    pub fn inverse_inplace(self: *Self) void {
        if (self.is_affine()) {
            self.inverse_affine_inplace();
        } else {
            self.inverse_general_inplace();
        }
    }

    // NOTE: this should be known from the context
    // computing this explicitly would be expensive, though can be used in Debug mode
    pub fn is_orthogonal(self: Self) bool {
        const transp = self.transposed();
        const res = self.mul(transp);
        return res.is_identity();
    }

    inline fn is_affine(s: Self) bool {
        const affine_row: Vec4 = .init_from_slice(&.{ 0, 0, 0, 1 });
        const last_row_v = s.get_row(3);
        return last_row_v.cmp(affine_row);
    }

    // NOTE: faster than affine, so if you know that where was no scale on input matrix,
    // it's better to use this instead of affine
    ///////////////////////////////////////////////////////////////////////////////
    // compute the inverse of 4x4 Euclidean transformation matrix
    //
    // Euclidean transformation is translation, rotation, and reflection.
    // With Euclidean transform, only the position and orientation of the object
    // will be changed. Euclidean transform does not change the shape of an object
    // (no scaling). Length and angle are reserved.
    //
    // Use inverseAffine() if the matrix has scale and shear transformation.
    //
    // M = [ R | T ]
    //     [ --+-- ]    (R denotes 3x3 rotation/reflection matrix)
    //     [ 0 | 1 ]    (T denotes 1x3 translation matrix)
    //
    // y = M*x  ->  y = R*x + T  ->  x = R^-1*(y - T)  ->  x = R^T*y - R^T*T
    // (R is orthogonal,  R^-1 = R^T)
    //
    //  [ R | T ]-1    [ R^T | -R^T * T ]    (R denotes 3x3 rotation matrix)
    //  [ --+-- ]   =  [ ----+--------- ]    (T denotes 1x3 translation)
    //  [ 0 | 1 ]      [  0  |     1    ]    (R^T denotes R-transpose)
    ///////////////////////////////////////////////////////////////////////////////
    pub fn inverse_euclidean_inplace(self: *Self) void {
        // transpose 3x3 rotation matrix part
        // | R^T | 0 |
        // | ----+-- |
        // |  0  | 1 |
        std.mem.swap(f32, self.get_at_ptr(0, 1), self.get_at_ptr(1, 0));
        std.mem.swap(f32, self.get_at_ptr(0, 2), self.get_at_ptr(2, 0));
        std.mem.swap(f32, self.get_at_ptr(1, 2), self.get_at_ptr(2, 1));

        // compute translation part -R^T * T
        // | 0 | -R^T x |
        // | --+------- |
        // | 0 |   0    |

        // float x = m[12];
        // float y = m[13];
        // float z = m[14];
        // m[12] = -(m[0] * x + m[4] * y + m[8] * z);
        // m[13] = -(m[1] * x + m[5] * y + m[9] * z);
        // m[14] = -(m[2] * x + m[6] * y + m[10]* z);
        const sub_mat: Matrix3x3 = .init_from_mat4x4(self);
        var translate_col = sub_mat.mul_vec3(self.get_translation_col());
        translate_col.negate_inplace();
        self.set_translation_col(translate_col);
    }

    // NOTE: faster than inverse_general, but slower than inverse_euclidean
    // should be used if matrix has scale
    ///////////////////////////////////////////////////////////////////////////////
    // compute the inverse of a 4x4 affine transformation matrix
    //
    // Affine transformations are generalizations of Euclidean transformations.
    // Affine transformation includes translation, rotation, reflection, scaling,
    // and shearing. Length and angle are NOT preserved.
    // M = [ R | T ]
    //     [ --+-- ]    (R denotes 3x3 rotation/scale/shear matrix)
    //     [ 0 | 1 ]    (T denotes 1x3 translation matrix)
    //
    // y = M*x  ->  y = R*x + T  ->  x = R^-1*(y - T)  ->  x = R^-1*y - R^-1*T
    //
    //  [ R | T ]-1   [ R^-1 | -R^-1 * T ]
    //  [ --+-- ]   = [ -----+---------- ]
    //  [ 0 | 1 ]     [  0   +     1     ]
    ///////////////////////////////////////////////////////////////////////////////
    // NOTE: formula: https://stackoverflow.com/questions/2624422/efficient-4x4-matrix-inverse-affine-transform
    pub fn inverse_affine_inplace(self: *Self) void {
        // R^-1
        var sub_mat: Matrix3x3 = .init_from_mat4x4(self);
        sub_mat.inverse_inplace();

        self.set_row_from_vec3(sub_mat.get_row(0), 0);
        self.set_row_from_vec3(sub_mat.get_row(1), 1);
        self.set_row_from_vec3(sub_mat.get_row(2), 2);

        // -R^-1 * T
        var translate_col = sub_mat.mul_vec3(self.get_translation_col());
        translate_col.negate_inplace();
        self.set_translation_col(translate_col);
    }

    // NOTE: this is slowest version of invert,
    // only for matrices which have a projection row non empty (!= [0, 0, 0, 1])
    // use inverse_affine of inverse_euclidean if you can
    ///////////////////////////////////////////////////////////////////////////////
    // compute the inverse of a general 4x4 matrix using Cramer's Rule
    // If cannot find inverse, return indentity matrix
    // M^-1 = adj(M) / det(M)
    ///////////////////////////////////////////////////////////////////////////////
    pub fn inverse_general_inplace(self: *Self) void {
        var m = self.as_single_slice();
        // get cofactors of minor matrices
        const cofactor0: f32 = cofactor(m[5], m[6], m[7], m[9], m[10], m[11], m[13], m[14], m[15]);
        const cofactor1: f32 = cofactor(m[4], m[6], m[7], m[8], m[10], m[11], m[12], m[14], m[15]);
        const cofactor2: f32 = cofactor(m[4], m[5], m[7], m[8], m[9], m[11], m[12], m[13], m[15]);
        const cofactor3: f32 = cofactor(m[4], m[5], m[6], m[8], m[9], m[10], m[12], m[13], m[14]);

        // get determinant
        const det: f32 = m[0] * cofactor0 - m[1] * cofactor1 + m[2] * cofactor2 - m[3] * cofactor3;
        if (@abs(det) <= std.math.floatEps(f32)) {
            self.identity_inplace();
            return;
        }

        // get rest of cofactors for adj(M)
        const cofactor4: f32 = cofactor(m[1], m[2], m[3], m[9], m[10], m[11], m[13], m[14], m[15]);
        const cofactor5: f32 = cofactor(m[0], m[2], m[3], m[8], m[10], m[11], m[12], m[14], m[15]);
        const cofactor6: f32 = cofactor(m[0], m[1], m[3], m[8], m[9], m[11], m[12], m[13], m[15]);
        const cofactor7: f32 = cofactor(m[0], m[1], m[2], m[8], m[9], m[10], m[12], m[13], m[14]);

        const cofactor8: f32 = cofactor(m[1], m[2], m[3], m[5], m[6], m[7], m[13], m[14], m[15]);
        const cofactor9: f32 = cofactor(m[0], m[2], m[3], m[4], m[6], m[7], m[12], m[14], m[15]);
        const cofactor10: f32 = cofactor(m[0], m[1], m[3], m[4], m[5], m[7], m[12], m[13], m[15]);
        const cofactor11: f32 = cofactor(m[0], m[1], m[2], m[4], m[5], m[6], m[12], m[13], m[14]);

        const cofactor12: f32 = cofactor(m[1], m[2], m[3], m[5], m[6], m[7], m[9], m[10], m[11]);
        const cofactor13: f32 = cofactor(m[0], m[2], m[3], m[4], m[6], m[7], m[8], m[10], m[11]);
        const cofactor14: f32 = cofactor(m[0], m[1], m[3], m[4], m[5], m[7], m[8], m[9], m[11]);
        const cofactor15: f32 = cofactor(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);

        // build inverse matrix = adj(M) / det(M)
        // adjugate of M is the transpose of the cofactor matrix of M
        const inv_det: f32 = 1.0 / det;
        m[0] = inv_det * cofactor0;
        m[1] = -inv_det * cofactor4;
        m[2] = inv_det * cofactor8;
        m[3] = -inv_det * cofactor12;

        m[4] = -inv_det * cofactor1;
        m[5] = inv_det * cofactor5;
        m[6] = -inv_det * cofactor9;
        m[7] = inv_det * cofactor13;

        m[8] = inv_det * cofactor2;
        m[9] = -inv_det * cofactor6;
        m[10] = inv_det * cofactor10;
        m[11] = -inv_det * cofactor14;

        m[12] = -inv_det * cofactor3;
        m[13] = inv_det * cofactor7;
        m[14] = -inv_det * cofactor11;
        m[15] = inv_det * cofactor15;
    }

    // NOTE: fast video https://www.youtube.com/watch?v=NEOqegcMw-Q
    pub fn determinant(self: Self) f32 {
        const m = self.as_single_vec();
        return m[0] * cofactor(m[5], m[6], m[7], m[9], m[10], m[11], m[13], m[14], m[15]) -
            m[1] * cofactor(m[4], m[6], m[7], m[8], m[10], m[11], m[12], m[14], m[15]) +
            m[2] * cofactor(m[4], m[5], m[7], m[8], m[9], m[11], m[12], m[13], m[15]) -
            m[3] * cofactor(m[4], m[5], m[6], m[8], m[9], m[10], m[12], m[13], m[14]);
    }

    // cofactor of 3x3 submatrix
    fn cofactor(m0: f32, m1: f32, m2: f32, m3: f32, m4: f32, m5: f32, m6: f32, m7: f32, m8: f32) f32 {
        return m0 * (m4 * m8 - m5 * m7) -
            m1 * (m3 * m8 - m5 * m6) +
            m2 * (m3 * m7 - m4 * m6);
    }

    pub fn init_look_at(pos: Vec3, target: Vec3, world_up: Vec3) Self {
        const dir = target.distance(pos);
        const right = world_up.cross(dir);
        const up = dir.cross(right);

        var res: Self = .init_identity();
        res.set_row(right, 0);
        res.set_row(up, 1);
        res.set_row(dir, 2);

        res.translate(pos.negated());

        return res;
    }

    // NOTE: article reference https://www.kdab.com/projection-matrices-with-vulkan-part-2/
    pub fn init_projection(t: f32, b: f32, l: f32, r: f32, n: f32, f: f32) Self {
        var res: Self = .zero;
        res.set_at((2.0 * n) / (r - l), 0, 0);
        res.set_at(-((r + l) / (r - l)), 0, 2);
        res.set_at((2.0 * n) / (b - t), 1, 1);
        res.set_at(-((b + t) / (b - t)), 1, 2);
        res.set_at(f / (f - n), 2, 2);
        res.set_at(-((n * f) / (f - n)), 2, 3);
        res.set_at(1.0, 3, 2);
        return res;
    }

    pub fn init_projection_fov(fov: f32, aspect: f32, n: f32, f: f32) Self {
        var res: Self = .zero;
        const fov_half = fov / 2;
        const tan = @tan(fov_half);
        const b = tan * n;
        const r = b * aspect;

        res.set_at(n / r, 0, 0);
        res.set_at(n / b, 1, 1);
        res.set_at(f / (f - n), 2, 2);
        res.set_at(-((n * f) / (f - n)), 2, 3);
        res.set_at(1.0, 3, 2);

        return res;
    }

    pub inline fn set_row(self: *Self, input_row: Vec4, index: usize) void {
        self.m[index] = input_row;
    }

    pub inline fn set_row_from_vec3(self: *Self, input_row: Vec3, index: usize) void {
        self.m[index].set_from_vec(3, input_row);
    }

    pub inline fn set_row_from_vec2(self: *Self, input_row: Vec2, index: usize) void {
        self.m[index].set_from_vec(2, input_row);
    }

    pub inline fn increment_row(self: *Self, input_row: Vec4, index: usize) void {
        self.m[index].v += input_row.v;
    }

    pub inline fn set_col(self: *Self, input_col: Vec4, index: usize) void {
        for (0..ROWS) |i| {
            self.set_at(input_col.v[i], i, index);
        }
    }

    pub inline fn set_col_from_vec3(self: *Self, input_col: Vec3, index: usize) void {
        for (0..3) |i| {
            self.set_at(input_col.v[i], i, index);
        }
    }

    pub inline fn set_translation_col(self: *Self, input_col: Vec3) void {
        for (0..3) |i| {
            self.set_at(input_col.v[i], i, 3);
        }
    }

    pub inline fn increment_col(self: *Self, input_col: Vec4, index: usize) void {
        for (0..ROWS) |i| {
            self.increment_at(input_col.v[i], i, index);
        }
    }

    pub inline fn increment_col_from_vec3(self: *Self, input_col: Vec3, index: usize) void {
        for (0..3) |i| {
            self.increment_at(input_col.v[i], i, index);
        }
    }

    pub inline fn get_row(self: *const Self, index: usize) Vec4 {
        return self.m[index];
    }

    pub inline fn get_row_as_vec3(self: *const Self, index: usize) Vec3 {
        return vec4_to_vec3(self.m[index]);
    }

    pub fn get_col(self: *const Self, index: usize) Vec4 {
        var col: Vec4 = .{};
        for (0..ROWS) |i| {
            col.v[i] = self.m[i].v[index];
        }
        return col;
    }

    pub fn get_col_as_vec3(self: *const Self, index: usize) Vec3 {
        var col: Vec3 = undefined;
        for (0..3) |i| {
            col.v[i] = self.get_at(i, index);
        }
        return col;
    }

    pub fn get_scaling_vec(self: *const Self) Vec3 {
        var scale_vec: Vec3 = undefined;
        scale_vec.v[0] = self.get_at(0, 0);
        scale_vec.v[1] = self.get_at(1, 1);
        scale_vec.v[2] = self.get_at(2, 2);
        return scale_vec;
    }

    pub fn get_translation_col(self: *const Self) Vec3 {
        var col: Vec3 = .{};
        for (0..3) |i| {
            col.v[i] = self.get_at(i, 3);
        }
        return col;
    }

    pub inline fn set_at(self: *Self, val: f32, row: usize, col: usize) void {
        self.m[row].v[col] = val;
    }

    pub inline fn increment_at(self: *Self, val: f32, row: usize, col: usize) void {
        self.m[row].v[col] += val;
    }

    pub inline fn get_at(self: *const Self, row: usize, col: usize) f32 {
        return self.m[row].v[col];
    }

    // shorter alias for get_at
    pub inline fn at(self: *const Self, row: usize, col: usize) f32 {
        return self.m[row].v[col];
    }

    pub inline fn get_at_ptr(self: *Self, row: usize, col: usize) *f32 {
        return &self.m[row].v[col];
    }

    pub inline fn get_at_ptr_as_const(self: *const Self, row: usize, col: usize) *const f32 {
        return &self.m[row].v[col];
    }

    pub inline fn as_single_vec(self: *Self) *@Vector(ITEMS, f32) {
        return @ptrCast(@alignCast(&self.m));
    }

    pub inline fn as_single_vec_as_const(self: *const Self) *const @Vector(ITEMS, f32) {
        return @ptrCast(@alignCast(&self.m));
    }

    pub inline fn as_single_slice(self: *Self) []f32 {
        return @ptrCast(@alignCast(&self.m));
    }

    pub fn cmp(lhs: Self, rhs: Self) bool {
        for (0..ROWS) |y| {
            const l_row = lhs.get_row(y);
            const r_row = rhs.get_row(y);
            if (!l_row.cmp(r_row)) {
                return false;
            }
        }
        return true;
    }

    pub fn print(self: Self) void {
        for (0..ROWS) |y| {
            self.get_row(y).print();
        }
    }
};

test {
    {
        logger.println("Translation: ", .{});
        var m: Matrix4x4 = .init_identity();
        m.set_translate_inplace(.init_dupl(2.0));
        m.print();
        logger.println("Transposition: ", .{});
        m.transpose_inplace();
        m.print();
    }
    {
        logger.println("Rotation: ", .{});
        var m: Matrix4x4 = .init_identity();
        m.rotate_inplace(90, .Z);
        m.print();
        logger.print_newline();
        m.rotate_inplace(-90, .Z);
        m.print();
    }
    {
        logger.println("Scaling: ", .{});
        var m: Matrix4x4 = .init_identity();
        m.print();
        logger.print_newline();
        m.set_scale_inplace(.init_dupl(2.0));
        m.print();
    }
    {
        logger.println("Mul many", .{});
        var m: Matrix4x4 = .init_identity();
        const m2: Matrix4x4 = .init_scaling(.init_dupl(2.0));
        const m3: Matrix4x4 = .init_scaling(.init_dupl(3.0));
        const m4: Matrix4x4 = .init_scaling(.init_dupl(4.0));
        const res = m.mul_many(&.{ m2, m3, m4 });
        res.print();
    }
    {
        logger.println("Projection", .{});
        const m: Matrix4x4 = .init_projection_fov(45.0, 1.2, 0.1, 100.0);
        m.print();
    }
    {
        logger.println("Conversion m4x4 to m3x3: ", .{});
        const m4: Matrix4x4 = .init_identity();
        const m3: Matrix3x3 = .init_from_mat4x4(&m4);
        m3.print();
    }
    {
        logger.println("Conversion m3x3 to m4x4: ", .{});
        const m3: Matrix3x3 = .init_identity();
        const m4: Matrix4x4 = .init_from_mat3x3(&m3);
        m4.print();
    }
    {
        logger.println("Invert - rotation ", .{});
        var m: Matrix4x4 = .init_rotation(45.0, .Z);
        m.print();

        logger.println("after: ", .{});

        m.inverse_affine_inplace();
        m.print();
    }
    {
        logger.println("Invert - scaling + translation ", .{});
        var m: Matrix4x4 = .init_scaling(.init_from_slice(&.{ 2, 2, 2 }));
        const t: Matrix4x4 = .init_translation(.init_from_slice(&.{ 4, 2, 1 }));
        m.mul_inplace(t);
        m.print();

        logger.println("after: ", .{});

        m.inverse_affine_inplace();
        m.print();
    }
}
