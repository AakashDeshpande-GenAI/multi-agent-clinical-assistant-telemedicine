# 🩺 Swasthya-Agent: Multi-Agent Telemedicine Clinical Copilot
**A Voice-Enabled, Privacy-First Clinical Triage Assistant powered by Google ADK 2.0 & Gemini 2.5 Flash**

[![Smart India Hackathon 2026](https://img.shields.io/badge/SIH-2026_Student_Innovation-orange.svg)](https://www.sih.gov.in/)
[![Google ADK 2.0](https://img.shields.io/badge/Google_ADK-2.0.0-blue.svg)](https://adk.dev/)
[![Gemini 2.5 Flash](https://img.shields.io/badge/Gemini-2.5_Flash-green.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Swasthya-Agent is an offline-first, multilingual clinical triage assistant designed for rural Primary Health Centres (PHCs), ASHA workers, and telemedicine kiosks across India.

## 🌟 Key Features
- **Multilingual Voice-First Ingestion:** Speech-to-Text in Hindi, Marathi, Tamil, Bengali, and English, with support for code-mixed **Hinglish and Minglish**.
- **Zero-Trust On-Device PII Scrubbing:** Local pre-flight regex sanitization removes Aadhar numbers, names, and phone numbers before data leaves the kiosk (DPDP Act compliant).
- **Sequential Multi-Agent Architecture:** Uses Google ADK 2.0's `SequentialAgent` to separate diagnostic extraction from clinical contradiction auditing.
- **Offline AYUSH & Primary Care RAG:** Embedded local knowledge base (`repertory.json`) querying Homeopathy, Ayurveda, and Primary Care protocols without external dependencies.
- **Human-in-the-Loop (HITL) Doctor Dashboard:** Enforces a mandatory clinical authorization gate before prescription generation.
- **Regional Audio Synthesis (TTS):** Speaks approved care instructions out loud in the patient's native Indian language.
- **Agent Telemetry & Observability:** Real-time logging of tool calls and state mutations via the ADK event stream.

## 🏗️ Architecture Flow
```text
[Patient Voice/Text] ──▶ [PII Redactor] ──▶ [Diagnostic Agent] ──▶ [Local RAG] ──▶ [HITL Gate] ──▶ [Critic Agent] ──▶ [Markdown Report + Regional TTS]