# ACP Multi-Agent Orchestration Demo

A comprehensive demonstration of multi-agent orchestration using the **Agent Communication Protocol (ACP)** to coordinate specialized AI agents for automated business workflows.

## 🚀 What is ACP?

The **Agent Communication Protocol (ACP)** is an open standard developed under the Linux Foundation that enables seamless communication between AI agents across different frameworks, teams, and infrastructures. 

### Key Features:
- **Universal Bridge**: Connects agents across different AI frameworks and technology stacks
- **RESTful API**: Standardized communication using REST endpoints
- **Multi-Modal Support**: Handles all forms of data (text, images, audio, etc.)
- **Flexible Operations**: Supports both synchronous and asynchronous communication
- **Agent Discovery**: Online and offline discovery of available agents
- **Streaming & Long-Running**: Handles real-time streaming and long-running tasks

Learn more at: [agentcommunicationprotocol.dev](https://agentcommunicationprotocol.dev)

## 📋 Demo Overview

This demo showcases a **multi-agent orchestration system** that coordinates three specialized agents to handle complex business workflows:

### Agents in the Demo:

1. **📧 Email Agent** (`langchain_agent.py`) - Port 8003
   - Processes incoming emails using LangGraph and MCP
   - Retrieves, categorizes, and manages email communications
   - Sends automated responses and confirmations

2. **📅 Calendar Agent** (`lg_calendar_agent.py`) - Port 8004  
   - Checks calendar availability for meeting requests
   - Suggests alternative times when conflicts exist
   - Provides availability confirmations

3. **🔍 Fraud Detection Agent** (`sm_agent.py`) - Port 8001
   - Analyzes email addresses for fraudulent patterns
   - Uses CodeAgent framework for intelligent fraud detection
   - Helps identify and flag suspicious communications

### 🎯 Orchestration Agent

The **ACP Calling Agent** (`fastacp.py`) acts as the coordinator:
- Receives complex user requests
- Determines which specialized agents to call
- Coordinates multi-step workflows
- Provides comprehensive final answers

## 🛠️ Quick Start

### Prerequisites

```bash
# Option 1: Install from requirements.txt (Recommended)
pip install -r requirements.txt


