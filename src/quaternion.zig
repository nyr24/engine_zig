// NOTE: implementation example https://github.com/kooparse/zalgebra/blob/main/src/quaternion.zig
// long article: https://lisyarus.github.io/blog/posts/introduction-to-quaternions.html#section-towards-rotations
// video explanation: https://www.youtube.com/watch?v=xoTqBqQtrY4

const std = @import("std");
const math = std.math;
const Matrix4x4 = @import("matrix.zig").Matrix4x4;
const vec = @import("vec.zig");
const logger = @import("logger.zig");
const Vec3 = vec.Vec3;
const Vec4 = vec.Vec4;

// In-Memory representation: (i, j, k, w)
pub const Quaternion = struct {
    const Self = @This();
    const LEN = 4;
    pub const identity = Self{.data{ 1.0, 0.0, 0.0, 0.0 }};

    data: Vec4,

    pub fn init_identity() Self {
        return identity;
    }

    pub fn init(x: f32, y: f32, z: f32, w: f32) Self {
        return .{ .data = .init(x, y, z, w) };
    }

    pub fn init_from_axis_angle(axis: Vec3, angle_deg: f32) Self {
        const angle_rad = math.degreesToRadians(angle_deg);
        const cos_half = @cos(angle_rad) * 0.5;
        const sin_half = @sin(angle_rad) * 0.5;
        const v_part = axis.mul_scalar(sin_half);
        return .{ .data = .init(v_part.data[0], v_part.data[1], v_part.data[2], cos_half) };
    }

    pub fn init_from_slice(slice: *const [LEN]f32) Self {
        return .{ .data = .init_from_slice(slice) };
    }

    pub fn init_from_vec4(input_v: Vec4) Self {
        return .{ .data = input_v };
    }

    pub fn init_from_vec3(input_v: Vec3) Self {
        return .{ .data = .init(input_v.x(), input_v.y(), input_v.z(), 0.0) };
    }

    pub fn inversed(self: Self) Self {
        var res: Self = self;
        res.inverse_inplace();
        return res;
    }

    pub fn inverse_inplace(self: *Self) void {
        const mag_squared_inv = 1 / self.magnitude_squared();
        self.conjugate_inplace();
        self.data.mul_inplace(mag_squared_inv);
    }

    pub fn conjugated(self: Self) Self {
        var res: Self = self;
        res.conjugate_inplace();
        return res;
    }

    pub fn conjugate_inplace(self: *Self) void {
        self.data.as_vec3().negate_inplace();
    }

    pub inline fn magnitude(self: Self) f32 {
        const mag_squared = self.magnitude_squared();
        return @sqrt(mag_squared);
    }

    pub inline fn magnitude_squared(self: Self) f32 {
        return self.dot(self);
    }

    pub inline fn dot(lhs: Self, rhs: Self) f32 {
        return lhs.data.dot(rhs.data);
    }

    pub fn normalized(self: Self) Self {
        var res: Self = self;
        res.normalize_inplace();
        return res;
    }

    pub fn normalize_inplace(self: *Self) void {
        const mag_inv = 1 / self.magnitude();
        self.data.mul_inplace(mag_inv);
    }

    pub fn mul_quat(lhs: Self, rhs: Self) Self {
        var res: Self = lhs;
        res.mul_quat_inplace(rhs);
        return res;
    }

    // returns result of combination in lhs
    pub fn mul_quat_inplace(lhs: *Self, rhs: Self) void {
        var lhs_v: *Vec3 = lhs.get_v();
        const rhs_v = rhs.get_v();
        const lhs_w = lhs.get_w();
        const rhs_w = rhs.get_w();
        lhs.set_w(lhs_w * rhs_w - lhs_v.dot(rhs_v));
        lhs.set_v(rhs_v.mul_scalar(lhs_w) + lhs_v.mul_scalar(rhs_w) + lhs_v.cross(rhs_v));
    }

    // it assumes that 'self' is normalized
    pub fn rotate_normalized(self: Self, point: Vec3) Vec3 {
        const point_quat = Self.init_from_vec3(point);
        const conj = self.conjugated();
        return self.mul(point_quat).mul(conj);
    }

    // less efficient when 'rotate_normalized'
    pub fn rotate(self: Self, point: Vec3) Vec3 {
        const self_normalized = self.normalized();
        const conj = self_normalized.conjugated();
        const point_quat = Self.init_from_vec3(point);
        return self.mul(point_quat).mul(conj);
    }

    pub fn to_rotation_matrix_from_normalized(self: Self) Matrix4x4 {
        const angle_rad = self.get_angle();
        const angle_deg = math.radiansToDegrees(angle_rad);
        const axis = self.get_axis_from_normalized();
        return .init_rotation_arbitrary_axis(angle_deg, axis);
    }

    // normalization will occur
    pub fn to_rotation_matrix(self: Self) Matrix4x4 {
        const angle_rad = self.get_angle();
        const angle_deg = math.radiansToDegrees(angle_rad);
        const axis = self.get_axis();
        return .init_rotation_arbitrary_axis(angle_deg, axis);
    }

    pub inline fn get_angle(self: Self) f32 {
        return 2 * math.acos(self.as_angle_axis().w);
    }

    pub inline fn get_axis_from_normalized(self: Self) Vec3 {
        const angle_rad_half = self.get_angle() * 0.5;
        const sin_inv = 1 / @sin(angle_rad_half);
        return self.as_angle_axis_as_const().v.mul_scalar(sin_inv);
    }

    // normalization will occur
    pub inline fn get_axis(self: Self) Vec3 {
        const self_n = self.normalized();
        return self_n.get_axis_from_normalized();
    }

    inline fn get_v(self: *Self) *Vec3 {
        return self.data.as_vec3();
    }

    inline fn set_v(self: *Self, v3: Vec3) void {
        return self.data.set_from_vec(3, v3);
    }

    inline fn get_w(self: *Self) f32 {
        return self.data.w();
    }

    inline fn set_w(self: *Self, w: f32) void {
        self.data[3] = w;
    }

    pub fn add(lhs: Self, rhs: Self) Self {
        var res = lhs;
        res.add_inplace(rhs);
        return res;
    }

    pub inline fn add_inplace(lhs: *Self, rhs: Self) void {
        lhs.data.add(rhs.data);
    }

    pub fn sub(lhs: Self, rhs: Self) Self {
        var res = lhs;
        res.sub_inplace(rhs);
        return res;
    }

    pub fn sub_inplace(lhs: *Self, rhs: Self) void {
        lhs.data.sub(rhs.data);
    }

    pub fn eql(lhs: Self, rhs: Self) bool {
        return std.meta.eql(lhs, rhs);
    }

    pub fn print(self: Self) void {
        self.data.print();
    }
};

test "quat" {
    {
        logger.println("Conjugate: ");
        const q1: Quaternion = .init(1, 2, 3, 4);
        q1.print();
        const q_conj = q1.conjugated();
        q_conj.print();
    }
}
