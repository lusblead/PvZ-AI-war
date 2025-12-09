# Gemini GenAI Web App

这是一个基于 **React** 和 **Google GenAI SDK** 构建的 Web 应用程序。它允许用户直接在浏览器中与 Google 最新的 Gemini 人工智能模型进行交互。

## 🎯 项目功能

*   **智能文本生成**: 集成 `gemini-2.5-flash` 等最新模型，提供快速、高质量的文本问答。
*   **多模态支持**: (取决于具体实现) 支持文本、图像等多种输入形式。
*   **安全配置**: 支持用户端输入 API Key，确保密钥安全。

## 🚀 使用指南

### 1. 获取 API Key
在使用本应用之前，您需要从 Google AI Studio 获取 API 密钥：

1.  访问 [Google AI Studio](https://aistudio.google.com/)。
2.  登录您的 Google 账号。
3.  点击 **"Get API key"** 并创建一个新的密钥。
4.  *(注意)*: 某些高级模型功能可能需要绑定计费项目。

### 2. 启动与配置
1.  打开应用程序。
2.  如果系统提示输入 **API Key**，请粘贴您在第一步中获取的密钥。
3.  应用会自动连接到 Google 的服务器。

### 3. 进行交互
1.  **输入提示词**: 在文本框中输入您想问的问题、需要生成的代码或文章主题。
2.  **提交请求**: 点击生成按钮。
3.  **查看结果**: AI 的回复将实时显示在界面上。

## 🛠️ 技术栈

*   **Frontend**: React, TypeScript
*   **AI SDK**: [`@google/genai`](https://www.npmjs.com/package/@google/genai)
*   **Build Tool**: Vite (或相应构建工具)

## ⚠️ 注意事项

*   本应用直接在前端调用 Google API。虽然方便开发，但在生产环境中，建议确保存储 API Key 的安全性。
*   请确保您的网络环境可以访问 Google API 服务。
