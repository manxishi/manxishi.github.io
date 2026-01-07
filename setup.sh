#!/bin/bash

# Setup script for Hugo site with PaperMod theme

echo "Setting up Hugo site with PaperMod theme..."

# Check if Hugo is installed
if ! command -v hugo &> /dev/null; then
    echo "Hugo is not installed. Please install it first:"
    echo "  macOS: brew install hugo"
    echo "  Or visit: https://gohugo.io/installation/"
    exit 1
fi

# Initialize git submodule for PaperMod theme
if [ ! -d "themes/PaperMod" ]; then
    echo "Adding PaperMod theme as git submodule..."
    git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
else
    echo "PaperMod theme already exists. Updating..."
    git submodule update --init --recursive
fi

echo ""
echo "Setup complete!"
echo ""
echo "To preview your site locally, run:"
echo "  hugo server"
echo ""
echo "To build for production, run:"
echo "  hugo"
echo ""
echo "To add a new blog post, run:"
echo "  hugo new posts/your-post-title.md"

