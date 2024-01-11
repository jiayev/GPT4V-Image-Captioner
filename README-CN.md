# GPT4V-Image-Captioner / GPT4V图像打标器

[软件安装&演示视频](https://www.bilibili.com/video/BV1pw411g7X1/?spm_id_from=333.999.0.0&vd_source=22436c5073194cf38787049c34e04e02)

[英文版说明](https://github.com/jiayev/GPT4V-Image-Captioner/blob/main/README.md)


这是一款使用 Gradio 构建，可使用GPT-4-vision API或[cogVLM](https://github.com/THUDM/CogVLM)模型进行图像打标的多功能图像处理工具箱。特色功能包括：

- 一键安装及使用
- 单图反推及批量打标功能
- 云端GPT4 & 本地CogVLM双模型可选
- 可视化标签分析与处理
- 图像分桶预压缩
- 关键词筛查及水印图像识别

开发者: [Jiaye](https://civitai.com/user/jiayev1), [LEOSAM是只兔狲](https://civitai.com/user/LEOSAM), [SleeeepyZhou](https://space.bilibili.com/360375877), [Fok](https://civitai.com/user/fok3827), GPT4。 欢迎有兴趣的朋友加入，对本项目进行进一步的完善改进。


![下载](https://github.com/jiayev/GPT4V-Image-Captioner/assets/16369810/90612e2b-aac1-4368-84d6-482bb660f5aa)


# 安装和启动指南

### Windows（如自动安装失败，请参考[手动安装说明](#windows-手动安装说明)）

1. 以管理员权限打开命令提示符，并导航到您想要克隆仓库的目录。
2. 使用以下命令克隆仓库：
    ```
    git clone https://github.com/jiayev/GPT4V-Image-Captioner
    ```
3. 双击 `install_windows.bat` 运行，并安装所有必要的依赖项。
4. 安装完成后，您可以通过双击 `Start_windows.bat`来在终端中启动GPT4V-Image-Captioner。
5. 按住ctrl并点击终端中的URL地址（或复制URL地址在浏览器打开），将在默认浏览器中跳转打开Gradio应用界面。
6. 请在界面最上方输入OpenAI官方或者第三方的GPT-4V API Key与API Url，设置图像地址后，就可以图像打标了。


### Linux / macOS

1. 打开终端，并导航到您想要克隆仓库的目录。
2. 使用以下命令克隆仓库：
    ```
    git clone https://github.com/jiayev/GPT4V-Image-Captioner
    ```
3. 导航到克隆的目录：
    ```
    cd GPT4V-Image-Captioner
    ```
4. 使用以下命令使安装脚本和启动脚本变为可执行：
    ```
    chmod +x install_linux_mac.sh; chmod +x Start_linux_mac.sh
    ```
5. 执行安装脚本：
    ```
    ./install_linux_mac.sh
    ```
6. 在终端中执行启动脚本来启动GPT4V-Image-Captioner。
    ```
    ./Start_linux_mac.sh
    ```
7. 复制终端中显示的URL地址，在浏览器中打开Gradio应用界面。
8. 请在界面最上方输入OpenAI官方或者第三方的GPT-4V API Key与API Url，设置图像地址后，就可以图像打标了。


### Windows 手动安装说明

1. 按 `Win + R` 打开命令提示符。键入 `cmd` 然后按 `Enter` 。

2. 使用下面的命令克隆仓库至本地：
    ```
    git clone https://github.com/jiayev/GPT4V-Image-Captioner
    ```

3. 克隆完成后，切换到克隆的目录中：
    ```
    cd GPT4V-Image-Captioner
    ```

4. 在安装依赖库之前，在命令提示符中输入以下命令并按 `Enter` 来检查是否电脑已经安装了 Python：
    ```
    python --version
    ```
   如果未安装，会显示错误信息。请访问 [Python 官方下载页面](https://www.python.org/downloads/) 并按照指示进行安装。

5. 创建一个名为 `myenv` 的虚拟环境以避免污染全局 Python 环境：
    ```
    python -m venv myenv
    ```

6. 激活你刚创建的虚拟环境：
    ```
    myenv\Scripts\activate
    ```

7. 更新 `pip`至最新版本：
    ```
    python -m pip install --upgrade pip
    ```

8. 在虚拟环境中安装 `requests`、`gradio` 、 `tqdm` 等库：
    ```
    pip install scipy networkx wordcloud matplotlib Pillow tqdm gradio requests
    ```

9. 完成上述步骤后，可通过双击 `Start_windows.bat` 文件来启动 GPT4V-Image-Captioner。


## 更新内容

### 2024年1月6日
- **更智能的一键安装**: 增加了更智能的一键安装 (`install_windows.bat`) 功能，国内的小伙伴不用再看着pip十几kb慢慢爬了，更加国际化(×，简化了程序的安装。
- **CogVLM支持**: 增加了CogVLM模型的一键安装以及切换页面，没有GPT4的小伙伴也可以靠本地多模态快乐玩耍了（穷哥们狂喜。

### 2024年1月2日
- **一键安装和一键启动**: 增加了一键安装 (`install_windows.bat` / `install_linux_mac.sh`) 和一键启动 (`Start_windows.bat` / `Start_linux_mac.sh`) 功能，简化了程序的安装和启动过程。
- **环境说明补充**: 补充了在Windows和Linux环境下程序的安装和启动说明。

### 2024年1月1日
- **运行加速**: 提高了程序的打标速度。现在可以在2-3秒内完成一张图片的标注。
- **标签处理**: 对于已有标签的图像文件，提供了以下不同处理选项："覆盖", "前置插入", "结尾追加" 和 "跳过"。
- **子文件夹处理**: 新程序能够处理文件夹及其子文件夹中的所有图像文件，支持的图像格式包括：'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'。
- **程序中断**: 增加了在批量打标签过程中中断打标的功能。
- **报错筛查**: 可以根据关键词，将所有GPT标记失败的图像（例如NSFW内容）移动到新的文件夹中。
- **本地化**: 增加了对中文的支持。
