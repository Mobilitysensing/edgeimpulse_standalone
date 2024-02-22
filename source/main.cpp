#include <stdio.h>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"


// Callback function declaration
static int get_signal_data(size_t offset, size_t length, float *out_ptr);

// Raw features copied from test sample
static const float features[] = {
    -127, 144, 157, -144, 184, 178, -156, 235, 205, -139, 304, 226, -87, 355, 198, -24, 388, 109, 10, 385, 7, 16, 337, -84, 4, 280, -160, -49, 226, -198, -97, 195, -186, -112, 180, -189, -124, 172, -204, -127, 162, -210, -112, 139, -205, -108, 118, -186, -97, 88, -177, -70, 60, -157, -52, 43, -148, -37, 31, -159, -24, 28, -156, -25, 27, -166, -39, 12, -172, -49, -3, -160, -64, -19, -151, -78, -30, -124, -85, -33, -91, -96, -33, -63, -100, -19, -31, -97, -24, -3, -102, -37, 30, -97, -21, 51, -82, 0, 60, -67, 10, 67, -63, 24, 81, -58, 33, 82, -43, 54, 72, -27, 79, 64, -6, 78, 48, 18, 78, 24, 31, 87, 1, 46, 90, -16, 64, 100, -55, 78, 99, -111, 67, 84, -154, 34, 60, -169, 10, 24, -162, 3, 3, -142, 6, 3, -121, 15, 7, -123, 25, 12, -141, 34, 19, -165, 36, 22, -192, 34, 7, -208, 12, -4, -223, -34, -12, -241, -72, -27, -240, -97, -54, -213, -118, -78, -181, -133, -85, -153, -127, -79, -117, -111, -75, -82, -97, -67, -70, -88, -57, -51, -81, -51, -27, -72, -49, -22, -73, -42, -15, -67, -34, -3, -52, -42, 0, -48, -36, -6, -46, -30, -6, -24, -28, 1, -24, -15, 1, -40, -9, 3, -22, -13, 4, -22, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

int main(int argc, char **argv) {

    signal_t signal;            // Wrapper for raw input buffer
    ei_impulse_result_t result; // Used to store inference output
    EI_IMPULSE_ERROR res;       // Return code from inference

    // Calculate the length of the buffer
    size_t buf_len = sizeof(features) / sizeof(features[0]);

    // Make sure that the length of the buffer matches expected input length
    if (buf_len != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        ei_printf("ERROR: The size of the input buffer is not correct.\r\n");
        ei_printf("Expected %d items, but got %d\r\n",
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE,
                (int)buf_len);
        return 1;
    }

    // Assign callback function to fill buffer used for preprocessing/inference
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &get_signal_data;

    // Perform DSP pre-processing and inference
    res = run_classifier(&signal, &result, false);

    // Print return code and how long it took to perform inference
    ei_printf("run_classifier returned: %d\r\n", res);
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
    ei_printf("Visual anomalies:\r\n");
    for (uint32_t i = 0; i < result.visual_ad_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }
#endif

    return 0;
}

// Callback: fill a section of the out_ptr buffer when requested
static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (features + offset)[i];
    }

    return EIDSP_OK;
}
