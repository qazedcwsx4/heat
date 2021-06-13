// This file is intended to write a bitmap image file corresponding to
// the output produced from the heatedplate.c or heatedplate.f90
// program files.
//
// The BmpImage class was written by Daniel LePage, 2007
// The main method was written by Aaron Bloomfield, 2008

#include <iostream>
#include <fstream>
#include <vector>
#include "bmp_converter.h"

using std::vector;

class Color {
public:
    unsigned char r;
    unsigned char g;
    unsigned char b;

    Color() {};

    Color(double red, double green, double blue) :
            r(static_cast<unsigned char>(red * 255)),
            g(static_cast<unsigned char>(green * 255)),
            b(static_cast<unsigned char>(blue * 255)) {};
};

class BmpImage {
public:
    const int width;
    const int height;

    BmpImage(int width, int height);

    void writeToFile(const std::string &filename);

    void putPixel(int w, int h, Color c);

    ~BmpImage();

protected:
    Color **data;

    void writeInt(int value, std::ofstream &f);

    void writeHeader(std::ofstream &f);
};


/** Public methods ***/

BmpImage::BmpImage(int width, int height) : width(width), height(height) {
    data = new Color *[width];
    for (int i = 0; i < width; ++i) {
        data[i] = new Color[height];
    }
}

void BmpImage::writeToFile(const std::string &filename) {
    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::binary);
    writeHeader(outfile);

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            outfile << data[i][j].b;
            outfile << data[i][j].g;
            outfile << data[i][j].r;
        }

        for (int i = 0; i < (4 - (width * 3) % 4) % 4; i++)
            outfile << char(0);
    }
    outfile.close();
}

/** Private methods ***/

void BmpImage::putPixel(int w, int h, Color c) {
    data[w][h] = c;
}

void BmpImage::writeInt(int value, std::ofstream &f) {
    union {
        int intvalue;
        struct {
            char a, b, c, d;
        } c;
    } e;
    e.intvalue = value;

    f << e.c.a << e.c.b << e.c.c << e.c.d;
}

void BmpImage::writeHeader(std::ofstream &f) {
    f << "BM";
    writeInt(0, f);
    writeInt(0, f);
    writeInt(54, f);
    writeInt(40, f);
    writeInt(width, f);
    writeInt(height, f);
    f << char(1);
    f << char(0);
    f << char(24) << char(0);
    writeInt(0, f);
    writeInt(0, f);
    writeInt(2835, f);
    writeInt(2835, f);
    writeInt(0, f);
    writeInt(0, f);
}

BmpImage::~BmpImage() {
    for (int i = 0; i < width; ++i) {
        delete[] data[i];
    }
    delete[] data;
}

std::tuple<float, float> getMinMax(const std::string &filename) {
    std::ifstream inputFile;
    inputFile.open(filename.c_str());

    int height, width;

    inputFile >> height;
    inputFile >> width;
    float max = std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float number;
            inputFile >> number;

            if (number < min) {
                min = number;
            }
            if (number > max) {
                max = number;
            }
        }
    }
    inputFile.close();

    return {min, max};
}

Color hsvToColor(float h, float s, float v) {
    volatile double r;
    volatile double g;
    volatile double b;

    h /= 60;
    float k = floor(h);
    float f = h - k;
    float p = v * (1 - s);
    float q = v * (1 - (s * f));
    float t = v * (1 - (s * (1 - f)));
    if (k == 0) {
        r = v;
        g = t;
        b = p;
    } else if (k == 1) {
        r = q;
        g = v;
        b = p;
    } else if (k == 2) {
        r = p;
        g = v;
        b = t;
    } else if (k == 3) {
        r = p;
        g = q;
        b = v;
    } else if (k == 4) {
        r = t;
        g = p;
        b = v;
    } else if (k == 5) {
        r = v;
        g = p;
        b = q;
    }
    return {r, g, b};
}

int convert(const std::string &input, const std::string &output) {
    auto[min, max] = getMinMax(input);

    std::ifstream inputFile;
    inputFile.open(input.c_str());
    int height, width;

    inputFile >> height;
    inputFile >> width;

    BmpImage image(width, height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float d;
            inputFile >> d;

            float h = 240.0 * (max - d) / (max - min);

            Color color = hsvToColor(h, 1.0f, 1.0f);

            image.putPixel(j, height - i - 1, color);
        }
    }

    inputFile.close();
    image.writeToFile(output);
    return 0;
}

#pragma clang diagnostic pop