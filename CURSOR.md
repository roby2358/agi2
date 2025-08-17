# CURSOR Development Workflow

## Overview
We follow an iterative, requirements-driven development approach that emphasizes working code, comprehensive testing, and continuous improvement. This workflow integrates with our pjpd project management system to track progress and maintain focus.

## Core Development Flow

### 1. Task Management & Planning
- **Add/Update Tasks**: Use pjpd tools to create and manage development tasks
- **Prioritize**: Set clear priority levels (higher numbers = higher priority)
- **Track Progress**: Mark tasks as complete when requirements are met

### 2. Requirements Definition
- **Update SPEC.md**: Add detailed requirements for each task in MUST/SHOULD/MAY format
- **Define Interfaces**: Specify function signatures, data structures, and behavior
- **Document Constraints**: Note performance requirements, error handling, and edge cases

### 3. Open Questions & Research
- **Identify Uncertainties**: Document areas where requirements are unclear
- **Research Solutions**: Investigate best practices and implementation approaches
- **Document Decisions**: Record rationale for chosen solutions

### 4. Implementation
- **Follow Requirements**: Code closely to the specified requirements
- **Additive Changes**: Prefer adding new functionality over modifying existing code
- **Functional Approach**: Use functional programming constructs where appropriate
- **Keep It Simple**: Focus on solving the specific problem first

### 5. Testing (Critical)
- **Unit Tests Required**: Always write reusable unit tests in the test/ directory
- **Test Against Requirements**: Verify code behavior matches SPEC.md requirements
- **No Throwaway Scripts**: All test code must be maintainable and reusable
- **Test Coverage**: Aim for comprehensive coverage of new functionality

### 6. Review & Iteration
- **Code Review**: Assess implementation against requirements
- **Performance Analysis**: Identify areas for optimization
- **Task Updates**: Review existing tasks and add new ones as needed
- **Continuous Improvement**: Refactor and enhance based on learnings

## Development Principles

### Code Quality
- **Readability**: Write clear, self-documenting code
- **Maintainability**: Structure code for easy modification and extension
- **Consistency**: Follow established patterns and conventions

### Testing Philosophy
- **Test-First**: Consider testing requirements during design
- **Comprehensive**: Cover happy path, edge cases, and error conditions
- **Maintainable**: Tests should be easy to understand and modify
- **Fast**: Unit tests should run quickly for rapid feedback

### Project Management
- **Task Tracking**: Use pjpd for all development work
- **Priority Management**: Focus on high-priority tasks first
- **Progress Visibility**: Regular updates on task status and completion

## Workflow Commands

### Common pjpd Operations
```bash
# List current tasks and priorities
pjpd list_tasks --project agi2

# Add new development task
pjpd add_task --project agi2 --description "Implement feature X" --tag "feature-x" --priority 75

# Mark task complete
pjpd mark_done --project agi2 --task_id "feature-x-XXXX"

# Get next steps
pjpd next_steps --max_results 5
```

### Development Commands
```bash
# Run tests
python -m pytest tests/

# Check project structure
ls -la src/ tests/

# Update project dependencies
pip install -r requirements.txt
```

## File Organization

### Source Code
- `src/` - Main implementation code
- `tests/` - Unit tests (mirror src/ structure)
- `SPEC.md` - Detailed requirements and specifications
- `pyproject.toml` - Project configuration and dependencies

### Project Management
- `pjpd/` - Project task files and configuration
- `CURSOR.md` - This development workflow guide
- `README.md` - Project overview and setup instructions

## Getting Started

1. **Review Current Tasks**: Check pjpd for existing work
2. **Select Next Task**: Choose highest priority incomplete task
3. **Update Requirements**: Ensure SPEC.md has clear requirements
4. **Implement**: Code to requirements with tests
5. **Verify**: Write and run unit tests and validate against requirements
6. **Complete**: Mark task done and move to next priority

## Collaboration Notes

- **Cursor Integration**: Use Cursor's AI assistance for code review and suggestions
- **Version Control**: Commit working code with clear commit messages
- **Documentation**: Update docs as implementation progresses
- **Feedback Loop**: Regular review of workflow effectiveness

---

*This workflow ensures we build working, tested code that meets requirements while maintaining a clear development path and project visibility.*
