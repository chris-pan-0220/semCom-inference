#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include "onnxruntime_c_api.h"

int main(int argc, char* argv[]) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s -n SEQUENCE_LEN ( between 1-32 )\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "-n") != 0) {
        fprintf(stderr, "Error: missing argument `-n`\n");
        return 1;
    }

    int seq_len = atoi(argv[2]);

    if (seq_len < 1 || seq_len > 32) {
        fprintf(stderr, "Error:  argument `SEQUENCE_LEN` must be between 1-32\n");
        return 1;
    }
    
    /**************** initialize ****************/
    
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // create env
    OrtEnv* env;
    const OrtStatus *status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);

    // create create session options
    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);

    // create create session
    OrtSession* session;
    status = g_ort->CreateSession(env, "src/model/square.onnx", session_options, &session);

    /**************** inference ****************/

    // create input tensor values
    size_t input_tensor_size = seq_len;
    float* input_tensor_values = (float*)malloc(input_tensor_size * sizeof(float));

    // fill input_tensor_values: [1.0, 2.0, ...]
    for(size_t i = 0;i < input_tensor_size;i++){
        input_tensor_values[i] = i+1;
    }

    OrtMemoryInfo* memory_info;
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

    OrtValue* input_tensor = NULL;
    size_t shape_len = 1;
    int64_t *shape = (int64_t*)malloc(sizeof(int64_t)*shape_len);// input shape: (2)
    shape[0] = seq_len;

    status = g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values, input_tensor_size * sizeof(float), shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    // set input node & output node 
    const char* input_node_names[] = {"input"};
    const char* output_node_names[] = {"output"};
    
    // inference
    OrtValue* output_tensor = NULL;
    
    status = g_ort->Run(session, NULL, input_node_names, (const OrtValue* const*)&input_tensor, 1, output_node_names, 1, &output_tensor);
    
    // get output
    float* output_tensor_values;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_values);

    // return 0;
    
    // get output shape
    OrtTensorTypeAndShapeInfo *output_info = NULL;
    size_t output_shape_len = 0;
    int64_t *output_shape = (int64_t*)malloc(sizeof(int64_t) * output_shape_len);
    status = g_ort->GetTensorTypeAndShape(output_tensor, &output_info);
    status = g_ort->GetDimensionsCount(output_info, &output_shape_len);
    status = g_ort->GetDimensions(output_info, output_shape, output_shape_len);

    printf("input shape: ");
    printf("(");
    for(size_t i = 0;i < shape_len;i++){
        if(i != 0)
            printf(",");
        printf("%lu", shape[i]);
    }
    printf(")\n");

    printf("output shape: ");
    printf("(");
    for(size_t i = 0;i < output_shape_len;i++){
        if(i != 0)
            printf(",");
        printf("%ld", output_shape[i]);
    }
    printf(")\n");

    // it must be one-dimension output
    assert(output_shape_len == 1);

    printf("output: ");
    printf("(");
    for(size_t i = 0;i < output_shape[0];i++){
        if(i != 0)
            printf(", "); 
        printf("%f", output_tensor_values[i]);
    }
    printf(")\n");

    /**************** release ****************/

    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    free(input_tensor_values);
    free(shape);

    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
    free(output_shape);

    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    return 0;
}
