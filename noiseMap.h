#ifndef  __HEIGHTMAP_H__
#define  __HEIGHTMAP_H__

struct NoiseMap {

    NoiseMap(int w, int h) {
        width = w;
        height = h;
        data = new float[width * height];
    }

    void clear(float h) {

        int numPixels = width * height;
        float* ptr = data;
        for (int i=0; i<numPixels; i++) {
            ptr[0] = h;
            ptr += 1;
        }
    }

    int width;
    int height;
    float* data;
};


#endif