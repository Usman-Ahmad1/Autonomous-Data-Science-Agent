# 🤖 Autonomous Data Science Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0+-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20+-green.svg)](https://langchain.com/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 The Problem

**After spending countless hours repeating the same machine learning workflow for different datasets, I decided to build an AI-powered system that automates the complete tabular classification pipeline.**

## 📋 Overview

An **Autonomous Data Science Agent** that provides an end-to-end machine learning workflow for tabular classification and regression problems. Built with Streamlit, LangChain, and LangGraph, this application orchestrates multiple specialized agents to automate the entire data science pipeline from data cleaning to model selection.


## 🎯 Architecture & Workflow

### System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Streamlit UI]
        UPLOAD[File Upload]
        DISPLAY[Results Display]
    end
    
    subgraph "Orchestration Layer"
        LG[LangGraph Workflow]
        GRAPH[graph.py]
    end
    
    subgraph "Agent Layer"
        EDA[EDA Agent<br/>eda_agent.py]
        CLEAN[Cleaning Agent<br/>cleaning_agent.py]
        FEAT[Feature Agent<br/>feature_agent.py]
        MODEL[Model Agent<br/>model_agent.py]
    end
    
    subgraph "Core Modules"
        VIZ[Visualization<br/>src/visualization]
        FE[Feature Engineering<br/>src/feature_engineering]
        MS[Model Selection<br/>src/model_selection]
        CLEAN2[Cleaning Pipeline<br/>src/cleaning]
    end
    
    subgraph "Support Layer"
        PROMPTS[System Prompts<br/>prompts/system_prompts.py]
        TOOLS[Tools<br/>src/tools]
    end
    
    UI --> LG
    LG --> EDA
    LG --> CLEAN
    LG --> FEAT
    LG --> MODEL
    
    EDA --> VIZ
    CLEAN --> CLEAN2
    FEAT --> FE
    MODEL --> MS
    
    CLEAN --> PROMPTS
    FEAT --> PROMPTS
    MODEL --> PROMPTS
    
    MODEL --> TOOLS
