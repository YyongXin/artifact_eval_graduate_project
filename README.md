<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png">
</div>

# 毕业论文 Artifacts Evaluation

## 1.概述

	此代码用于我的毕业论文的工件评估，题目为“深度学习训练系统的GPU内存优化研究”。

## 2.本分支主要内容
- 代码仓库
	
	所有与METS相关的代码都可以从[https://github.com/YyongXin/tf-mets.git]访问得到。
- 技术文档:

   -- 文档：包括如何构建，安装，测试TF-METS的详细指南
   
   
   -- Benchmark：测试TF-METS的Benchmark模型代码
   
   
   -- 完整性：TF-METS的源码以及相关组件代码
   
   
   -- 图表脚本：论文中图表数据的复现脚本和相应原始数据

## 3.代码编译指南
**Step 1:**

从anaconda的base环境中创建新的环境：

```
$ conda create -n tf2-source python=3.6
$ conda activate tf2-source
```
最好选择`python3`，否则编译脚本会出现兼容性错误！！！

基于该虚拟环境进行从源码build。
build过程中如果缺少相关的依赖模块，手工安装。

不同版本的TensorFlow在`.bazelversion`中指定了Bazel的版本，所以需要手动安装特定版本的Bazel。

Bazel安装可以从源github中release页面下载installer-xxx.sh

```
$ ./bazel-0.29.1-installer-linux-x86_64.sh --user
```
安装在个人目录下，不需要sudo权限。
**step 2:**

克隆TF-METS原始代码,并进入项目目录下
```
$ git clone https://github.com/YyongXin/tf-mets.git
$ cd tf-mets
```
**Step 3:**

```
$ ./configure
```
选择需要的选项，tensorrt可以关掉。默认会打开XLA选项，它会依赖llvm，会导致编译的过程非常慢，16核的机器上，线程打满需要30分钟到1小时不等。

然后，`bazel build`进行编译。

gpu版本的tensorflow选项如下：

```
$ bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

安装的过程中，由于网络原因部分依赖库可能下载不下来，此时可以手工下载，放在提示的目录下，如：

```
ERROR: An error occurred during the fetch of repository 'io_bazel_rules_go':
   Traceback (most recent call last):
	File "/home/dataflow/.cache/bazel/_bazel_dataflow/a070875b19125f303ef2f02922aed5a5/external/bazel_tools/tools/build_defs/repo/http.bzl", line 111, column 45, in _http_archive_impl
		download_info = ctx.download_and_extract(
Error in download_and_extract: java.io.IOException: Error downloading [https://github.com/bazelbuild/rules_go/releases/download/0.18.5/rules_go-0.18.5.tar.gz] to /home/dataflow/.cache/bazel/_bazel_dataflow/a070875b19125f303ef2f02922aed5a5/external/io_bazel_rules_go/temp15150386869874428079/rules_go-0.18.5.tar.gz: connect timed out
```

上面信息提示缺少`rules_go-0.18.5.tar.gz`，此时可以从上述链接中下载下来并放在对应的`bazel cache`目录下，并修改成对应提示的文件名！
以上方式可以解决**国内网络不稳定**的问题！

对于CUDA 10.2 fatbinary选项错误，可以参考该[issue](https://github.com/tensorflow/tensorflow/issues/34429)删除`third_party/nccl/build_defs.bzl.tpl`
中的`"--bin2c-path=%s" % bin2c.dirname`

完成编译后源码目录下会生成若干个以bazel开头的文件夹：`bazel-out, bazel-bin, bazel-tensorflow-2.1.3, bazel-genfiles, bazel-testlogs`等。

编译python安装包：

```
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ pip install /tmp/tensorflow_pkg/tensorflow-2.1.3-cp35-cp35m-linux_x86_64.whl
```
安装完成后，不要在源码目录下测试，否则会有ImportError:

```
ImportError: Could not import tensorflow. Do not import tensorflow from its source directory; change directory to outside the TensorFlow source tree, and relaunch your Python interpreter from there.
```

**ERROR NOTES:**

bfloat16 error build with python 3.6, checkout this issue: <https://github.com/tensorflow/tensorflow/issues/40688>

- this mainly caused by numpy! downgrade numpy with `pip install numpy==1.18.0`,
- and remember execute `bazel clean`!!!

Upgrade python from python3.5 into python3.6 with conda, `conda install python3.6`,
reconfigure by `./configure`, and continues with normal building process. Note that, for different python version,
`./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg` will generate different pip wheel packages!
### REFs
- <https://www.tensorflow.org/install/source>
## 4.复现实验图表
论文中实验图表的复现脚本在图表对应文件夹下运行
```
$ cd figure.x.y
figure.x.y$ bash reproduce.sh 
```
运行脚本之后可以得到原文中图表
## 5.其他资源
*   [TensorFlow.org](https://www.tensorflow.org)
*   [TensorFlow tutorials](https://www.tensorflow.org/tutorials/)
*   [TensorFlow official models](https://github.com/tensorflow/models/tree/master/official)
*   [TensorFlow examples](https://github.com/tensorflow/examples)
*   [TensorFlow in Practice from Coursera](https://www.coursera.org/specializations/tensorflow-in-practice)
*   [Intro to TensorFlow for Deep Learning from Udacity](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
*   [Introduction to TensorFlow Lite from Udacity](https://www.udacity.com/course/intro-to-tensorflow-lite--ud190)
*   [TensorFlow blog](https://blog.tensorflow.org)
*   [TensorFlow Twitter](https://twitter.com/tensorflow)
*   [TensorFlow YouTube](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)
*   [TensorFlow roadmap](https://www.tensorflow.org/community/roadmap)
*   [TensorFlow white papers](https://www.tensorflow.org/about/bib)
*   [TensorBoard visualization toolkit](https://github.com/tensorflow/tensorboard)
## 6.文档
[TF-METS] TF-METS is a flexible and efficient deep learning training system that can train TensorFlow models with efficient memory usage efficiency, achieving high memory utilization and high efficiency.
[TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform
for machine learning. It has a comprehensive, flexible ecosystem of
[tools](https://www.tensorflow.org/resources/tools),
[libraries](https://www.tensorflow.org/resources/libraries-extensions), and
[community](https://www.tensorflow.org/community) resources that lets
researchers push the state-of-the-art in ML and developers easily build and
deploy ML powered applications.

TensorFlow was originally developed by researchers and engineers working on the
Google Brain team within Google's Machine Intelligence Research organization for
the purposes of conducting machine learning and deep neural networks research.
The system is general enough to be applicable in a wide variety of other
domains, as well.

TensorFlow provides stable [Python](https://www.tensorflow.org/api_docs/python)
and [C++](https://www.tensorflow.org/api_docs/cc) APIs, as well as
non-guaranteed backwards compatible API for
[other languages](https://www.tensorflow.org/api_docs).
