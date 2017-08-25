#pragma once

struct vec {
    double x;
    double y;

    vec& operator +=(const vec& rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    vec& operator *=(const vec& rhs) {
        x *= rhs.x;
        y *= rhs.y;
        return *this;
    }

    vec& operator -=(const vec& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    vec& operator /=(const vec& rhs) {
        x /= rhs.x;
        y /= rhs.y;
        return *this;
    }

    vec& operator *=(double rhs) {
        x *= rhs;
        y *= rhs;
        return *this;
    }

    vec& operator /=(double rhs) {
        x /= rhs;
        y /= rhs;
        return *this;
    }

    vec toCoords(const vec& a, double b);
};

vec operator +(vec lhs, const vec& rhs) {
    return lhs += rhs;
}

vec operator -(vec lhs, const vec& rhs) {
    return lhs -= rhs;
}

vec operator *(vec lhs, const vec& rhs) {
    return lhs *= rhs;
}

vec operator /(vec lhs, const vec& rhs) {
    return lhs /= rhs;
}

vec operator *(vec lhs, double rhs) {
    return lhs *= rhs;
}

vec operator /(vec lhs, double rhs) {
    return lhs /= rhs;
}

vec operator *(double lhs, vec rhs) {
    return rhs *= lhs;
}
