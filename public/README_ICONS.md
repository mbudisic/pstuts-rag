# 🎨 Chainlit Icon Customization Guide

This guide shows you how to customize the icons displayed in your Chainlit interface.

## 📁 File Structure

```
public/
├── theme.json              # Your existing theme
├── logo_light.png          # Logo for light mode (optional)
├── logo_dark.png           # Logo for dark mode (optional)
├── favicon.ico             # Browser tab icon (optional)
└── avatars/                # Message avatars (optional)
    ├── eva.png             # Eva AI assistant avatar
    ├── system.png          # System message avatar
    └── user.png            # User message avatar
```

## 🎯 Icon Specifications

### Logo Files
- **Size**: 200x200px recommended
- **Format**: PNG with transparent background
- **Naming**: Must be exactly `logo_light.png` and `logo_dark.png`

### Favicon
- **Size**: 32x32px or 64x64px
- **Format**: `.ico` (preferred) or `.png`
- **Naming**: Must be exactly `favicon.ico` or `favicon.png`

### Avatars
- **Size**: 64x64px recommended
- **Format**: PNG with transparent background
- **Naming**: Must match the author name in your code
  - `eva.png` for author="Eva"
  - `system.png` for author="System"
  - `user.png` for user messages

## 🚀 How to Apply Changes

1. **Add your icon files** to the `public/` folder
2. **Restart your Chainlit application**:
   ```bash
   chainlit run app.py
   ```
3. **Clear browser cache** if icons don't appear immediately

## 💡 Pro Tips

- Use **transparent backgrounds** for better integration
- Keep file sizes small for faster loading
- Test both light and dark modes
- Use consistent styling across all icons

## 🔍 Current Authors in Your App

Based on your `app.py`, you have these message authors:
- `"Eva"` - Your AI assistant
- `"System"` - System messages
- `"user"` - User messages (automatic)

Create corresponding avatar files for each!

