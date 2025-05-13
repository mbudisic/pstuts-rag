# 🛠️ Developer Documentation

## 📦 Project Structure

```
.
├── app.py              # Main Chainlit application
├── requirements.txt    # Project dependencies
└── README.md          # User documentation
```

## 🔧 Technical Details

The application uses Chainlit (v0.7.700+) to create a simple chat interface. The main functionality is implemented in `app.py` using the `@cl.on_message` decorator to handle incoming messages.

### Key Components

- `@cl.on_message`: Decorator that handles incoming messages from the chat interface
- `cl.Message`: Class for creating and sending messages back to the user
- Async/await pattern for handling message processing

## 🚀 Running Locally

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
chainlit run app.py
```

## 🔍 Debugging

- Chainlit provides a built-in debug mode. Run with:
```bash
chainlit run app.py --debug
```

## 📚 Resources

- [Chainlit Documentation](https://docs.chainlit.io)
- [Chainlit GitHub Repository](https://github.com/Chainlit/chainlit) 