# 🚀 5-Minute Technical Presentation: Enhanced Video Archive

## Setup:

- [github webpage](https://github.com/mbudisic/pstuts-rag)
- close all tabs
- 

## Introduction

My name is Marko Budisic, I work from Framatome in Lynchburg, VA, a couple of hours
north of us here.

##  Short Description

I am presenting here a work-in-progress on EVA, Enhanced Video Archive.
I worked on this as a part of my own learning about AI, through AI Makerspace - an excellent AI bootcamp program.

**So what's the use case we are tackling here?**

Many larger companies have recordings of internal training,
demos of their work processes, that they want to make usable for learning without 
users needing to watch hours of video.
EVA is intended to allow **knowledge extraction** from a library of non-annotated
videos, MP4s.

In my case, I am using a pre-made database of Adobe Photoshop training videos, 
as a stand-in for such a library.

## Demo

Let me give you a short demo.

## Under the hood

I built the app using LangGraph which allows us to show 
the AI workflow that the tool uses.

There are a few technical tricks that I made sure to build in:

- when using video transcripts as RAG context, exact timestamps are propagated 
  as metadata, so that the final response can link to the relevant time in a video,
  which is very useful for long video reviews
- since this is supposed to be an in-house tool, I made sure that all AI API calls
  can be rerouted to a local Ollama server, meaning that no information exits a 
  security-minded company
- additional guardrails: initial screening node, permissions before initiating web search

## Closing

Like I said, this is a work in progress. On the github page you can 
see that the next features I am going to build deal with multimodal search,
that is searching by screenshot, and by adding automated text extraction by
transcription and OCR, so I can stop relying on pre-transcripted videos.

Thank you very much and I'd love to chat more today or later, via LinkedIn.