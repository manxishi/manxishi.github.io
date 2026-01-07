# Manxi (Maggie) Shi - Personal Website

This is my personal website built with [Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

## Quick Setup

Run the setup script:

```bash
./setup.sh
```

This will check for Hugo installation and set up the PaperMod theme.

## Manual Setup Instructions

### 1. Install Hugo

If you don't have Hugo installed, you can install it:

**macOS (using Homebrew):**
```bash
brew install hugo
```

**Other platforms:**
Visit https://gohugo.io/installation/

### 2. Install PaperMod Theme

The theme is set up as a Git submodule. To initialize it:

```bash
git submodule update --init --recursive
```

If the submodule doesn't exist yet, add it:

```bash
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

### 3. Run Hugo Locally

To preview your site locally:

```bash
hugo server
```

Then visit `http://localhost:1313` in your browser.

### 4. Build for Production

To generate the static site:

```bash
hugo
```

This will create a `public/` directory with your static site.

### 5. Deploy to GitHub Pages

For GitHub Pages, you can:

1. **Option A: Use GitHub Actions** (Recommended)
   - Create `.github/workflows/hugo.yml` with Hugo deployment workflow
   - Push to GitHub and the site will build automatically

2. **Option B: Manual Deployment**
   - Run `hugo` locally
   - Copy contents of `public/` to your repository root
   - Push to GitHub

## Directory Structure

```
.
├── archetypes/      # Content templates
├── content/         # Your content (pages, posts, etc.)
│   ├── posts/       # Blog posts
│   └── ...
├── public/          # Generated static site (gitignored)
├── resources/       # Generated resources (gitignored)
├── themes/          # Hugo themes
│   └── PaperMod/    # PaperMod theme
├── config.yaml      # Hugo configuration
└── README.md        # This file
```

## Adding New Blog Posts

Create a new blog post:

```bash
hugo new posts/your-post-title.md
```

Or manually create a file in `content/posts/` with front matter like:

```yaml
---
title: "Your Post Title"
date: 2024-01-01
draft: false
description: "Post description"
tags: ["tag1", "tag2"]
---

Your content here...
```

## Customization

- Edit `config.yaml` to change site settings
- Add custom CSS in `assets/css/`
- Override theme templates in `layouts/`
