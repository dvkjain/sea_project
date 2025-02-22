# Sea project

The sea cli-tool is a simple way to read, modify, and do much more with spreadsheets using the terminal

# Installation

1. Clone the repository
2. Run the install.sh file

# Notes

- Functionality may be limited if a different format than .xlsx is being used, especially if editing a file.
- Only text can be read and edited from the commands (not images, animations, cell colors)

# Examples

```

sea read spreadsheet.xlsx

sea edit spreadsheet.xlsx A1 10

sea plot spreadsheet.xlsx line -x B1 -y C1 --saveimg graph.png

```
