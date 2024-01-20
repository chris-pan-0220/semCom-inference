# init

for simplicy, clone `mwnl` repo first 

```txt
repo
├── include
│   └── header files
├── lib
│   └── *.so, *.a
├── src
├── ... (Python scripts & *.c files)
```

# usage

use `make` to compile all *.out

```bash
make
```

# system latency solutions

考慮到inference需要在10ms以下，有以下解法：

### 1. C call Python interpreter

**方法**

在C當中直接使用system透過命令行操作python檔案，透過命令行參數傳遞input data

**問題**

啟動interpreter的速度非常的慢

### 2. C call Python function

**方法**

使用`Python.h`，在C當中直接call Python function，例如以下方法：

```C
// Generate by ChatGPT
#include <Python.h>

int main() {
    // 初始化Python解釋器
    Py_Initialize();

    // 加載Python模塊
    PyObject *pName = PyUnicode_FromString("my_ai_module");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "inference");
        
        // 確保函數存在且可調用
        if (pFunc && PyCallable_Check(pFunc)) {
            // 創建你的輸入參數
            PyObject *pArgs = PyTuple_New(2); // inference有2個參數
            PyTuple_SetItem(pArgs, 0, Py_BuildValue("O", ...)); // 設置每個參數，例如Py_BuildValue("i", 123)
            // ... 為其他參數賦值

            // 調用Python函數
            PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            // 處理返回值
            if (pValue != NULL) {
                // 處理Python函數的返回值
                Py_DECREF(pValue);
            } else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr, "Function call failed\n");
            }
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", "do_test");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", "my_ai_module");
    }

    // 清理Python解釋器
    Py_Finalize();
    return 0;
}
```

**問題**

在C當中，通常使用pointer指向數據儲存的空間；在Python當中則沒有pointer的概念，因此如果想要從C呼叫Python的function，例如以下：

```Python
def inference(
    input_data: torch.tensor,
    len_batch: torch.tensor
    # ...
):
    # ...
```

我們不能直接傳遞pointer給Python function，我們需要先使用`Py_BuildValue`來create物件，才能作為參數傳遞。

如果參數不是Python原生物件，例如`torch.tensor`，轉換會更麻煩。

### 3. using onnxruntime in C

**方法**

將encoder, decoder, normalize layer等需要用到的model，導出為onnx模型。

在C當中使用onnxruntime的API - `OrtApi`，可以直接執行model。

範例參考`square.c`

**問題**

在inference的過程中(參考`c_2_decoder_qpsk_gaussion.py`)，decode之後使用了`nltk`的BLEU score計算candidate sentence跟reference sentences之間的相似度，選出最相近的reference sentences。

這個部分就沒辦法匯出成onnx模型，讓C直接調用。

註：pytorch有計算BLEU score的API，可以嘗試把這個function包成一個單獨的pytorch model，然後匯出成onnx model讓C調用，但是這個解法並不通用，如果inference方法改變，C的部份就要重寫，會變得很複雜。

### 4. C Communicate with Python through unix socket

**方法**

在UE端，

C收到DCI之後，將訊息透過unix socket傳送給Python，Python將inference結果回傳給C。

如果需要傳輸的訊息是比較複雜的數據結構，例如多維度陣列、struct等，發送端需要先將數據序列化才能傳輸，接收端先反序列化後處理數據。

考慮到如果傳輸的數據如果較大，可以使用`protocol buffers`，這是一種比較高效的序列化/反序列化工具。

**benchmark for decoder - using cpu**

單獨測試decoder的性能，方法如下：

```bash
python3 decoder.py --n 10000
./testDecoder.out -n 10000
```

結果如下：

./testDecoder.out

```txt
9/100000 exceed 10ms
```

decoder.py

```txt
perf in ms:
              decode      inference           bleu           send          total
count  100000.000000  100000.000000  100000.000000  100000.000000  100000.000000
mean        6.771630       6.923114       0.189049       0.007539       7.119702
std         1.763395       1.787313       0.033177       0.003976       1.813019
min         2.263308       2.367973       0.094414       0.002861       2.476454
25%         7.340193       7.501364       0.195265       0.007391       7.706404
50%         7.361412       7.522583       0.196695       0.007629       7.728100
75%         7.380486       7.542372       0.198364       0.007868       7.748604
max       258.034468     258.504391       2.100468       0.395536     258.727789
```

* 使用cpu inference性能不穩定。

**benchmark for decoder - using gpu**

./testDecoder.out

```txt
0/100000 exceed 10ms
```

decoder.py

```txt
perf in ms:
              decode      inference           bleu           send          total
count  100000.000000  100000.000000  100000.000000  100000.000000  100000.000000
mean        2.813079       2.960023       0.104619       0.004213       3.068855
std         0.088674       0.093033       0.007986       0.000811       0.094761
min         2.661228       2.793789       0.089645       0.002146       2.897739
25%         2.780437       2.925634       0.102043       0.004053       3.033638
50%         2.808571       2.954841       0.103474       0.004053       3.063440
75%         2.839088       2.987146       0.105381       0.004292       3.096581
max         9.014368       9.345770       1.768351       0.030041       9.572268
```

* 需要注意的是，第一次inference的時候比較慢，需要讓模型空轉幾次，這稱作`warm-up`。
* 經過100000次測試，時間都在10ms之內
* BLEU考慮換成pytorch的API

**問題**
* 啟動時間比較久
* 如果Python的process掛掉怎麼辦 ?

**benchmark for encoder & decoder - using gpu**

先啟動這兩個：

```bash
python3 encoder.py --n 10000
python3 decoder.py --n 10000
```

再依順序啟動`recver.out`, `sender.out`

```bash
./recver.out -n 10000
./sender.out -n 10000
```

結果如下：

./sender.out

```txt
0/10000 exceed 10ms
```

./recver.out

```txt
0/10000 exceed 10ms
```

encoder.py

```txt
perf in ms:
             encode     inference          send         total
count  10000.000000  10000.000000  10000.000000  10000.000000
mean       0.541198      2.218740      0.024330      2.243070
std        0.073939      0.111207      0.003342      0.112325
min        0.468254      2.121925      0.018358      2.143621
25%        0.522614      2.196074      0.023365      2.220154
50%        0.532389      2.206326      0.023842      2.230406
75%        0.545979      2.219200      0.024319      2.243757
max        1.688957      4.410505      0.068903      4.469633
```

decoder.py

```txt
perf in ms:
             decode     inference          bleu          send         total
count  10000.000000  10000.000000  10000.000000  10000.000000  10000.000000
mean       2.937306      3.185578      0.111599      0.005091      3.302267
std        0.247173      0.255218      0.007605      0.000907      0.258669
min        2.770901      2.979994      0.101328      0.003815      3.088236
25%        2.896786      3.142357      0.108957      0.004768      3.257751
50%        2.921581      3.169298      0.110149      0.005007      3.285408
75%        2.947092      3.196478      0.111818      0.005245      3.314078
max        9.070158      9.536743      0.277996      0.030279      9.780407
```

# install onnxruntime for inference, cpu version

https://onnxruntime.ai/docs/build/inferencing.html

先檢查requirment：

* python 3.x
* cmake 3.17 or higher

接下來執行`./build.sh` to build from source

可以添加`--cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1`，以啟動debug模式，可以print inference的過程

```bash
git clone https://github.com/microsoft/onnxruntime
cd https://github.com/microsoft/onnxruntime
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
```

# using ortApi - onnxruntime api for C

引用以下標頭檔，以使用C的api：

```c
#include "onnxruntime_c_api.h"
```

編譯參數：
* `-I /path/to/onnxruntime_c_api.h`：`-I`指定header files搜尋路徑
* `-L /path/to/libonnxruntime.so -lonnxruntime`：`-L`指定動態函式庫的路徑，`-l`指定動態函式庫的名稱
    * 也可以透過`-Wl,-rpath=/path/to/libonnxruntime.so`指定動態函式庫的路徑

```bash
gcc -o main main.c -I /path/to/onnxruntime_c_api.h -L /path/to/libonnxruntime.so -lonnxruntime`
```

example：

```bash
gcc -o test test.c -I include -L lib -l onnxruntime  -Wl,-rpath=lib
```

# source

It's a patch for eric's repo, see https://github.com/eric693/mwnl