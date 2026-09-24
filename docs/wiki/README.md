# Wiki pages

These are the source files for the project's
[GitHub wiki](https://github.com/thomasmd321/AI_Programming_with_Python_Project_1/wiki).
Keeping them here means changes to them are reviewed and versioned with the code.

Start with [Home](Home.md). The pages link to each other with wiki-style
`[[Page Name]]` links, which work on the wiki but not in this folder.

## Publishing to the wiki

From the repository root, with your own GitHub login:

```bash
sh docs/wiki/publish.sh
```

It copies every page here (except this README) into the wiki and pushes.
Run it again whenever the pages change. `_Sidebar.md` becomes the wiki's
sidebar.
