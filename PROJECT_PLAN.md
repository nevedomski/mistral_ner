# Mistral NER Project Plan

## Overview
This document outlines the prioritized development plan for the Mistral NER project based on Linear issues analysis.

## Project Phases

### Phase 1: Code Quality Foundation (Week 1)

#### 1. NER-1: Ruff formatting and linting (HIGH PRIORITY)
- **Objective**: Establish consistent code quality foundation
- **Tasks**:
  - Add reasonable Ruff defaults to pyproject.toml
  - Lint and format whole codebase using Ruff
  - Replace Black and Flake8 with Ruff in GitHub Actions
- **Rationale**: Essential foundation for consistent code quality, simplifies tooling

#### 2. NER-2: Add MyPy (HIGH PRIORITY)
- **Objective**: Implement type safety across the project
- **Tasks**:
  - Add MyPy to project dependencies
  - Add type hint coverage for whole project
  - Add mypy check to GitHub workflow with Ruff
- **Rationale**: Type safety catches bugs early, improves maintainability

#### 3. NER-5: Test coverage (HIGH PRIORITY)
- **Objective**: Achieve 85%+ test coverage
- **Tasks**:
  - Fix existing unit tests
  - Add additional unit tests to cover 90%+ for every file
  - Ensure test job from GitHub actions finishes successfully
- **Rationale**: Currently tests are failing, essential before making model changes

### Phase 2: Infrastructure & Security (Week 2)

#### 4. NER-9: Add CodeQL scanning (MEDIUM PRIORITY)
- **Objective**: Implement security scanning
- **Tasks**:
  - Add CodeQL scanning to project
  - Configure security policies
- **Rationale**: GitHub security best practice, automated vulnerability detection

#### 5. NER-8: Fix offline wandb run (MEDIUM PRIORITY)
- **Objective**: Enable offline experimentation
- **Tasks**:
  - Fix offline wandb run functionality
  - Find a way to conveniently run project offline and later upload to wandb
- **Rationale**: Enables offline experimentation, important for distributed training

### Phase 3: Model Performance (Weeks 3-4)

#### 6. NER-3: Create better config file (HIGH PRIORITY)
- **Objective**: Improve F1 score from 0.8+
- **Tasks**:
  - Deep research on model parameters
  - Create new config file with optimized parameters
  - Validate improvements through experiments
- **Rationale**: Direct impact on model performance, foundation for further optimizations

#### 7. NER-6: Additional FineTuning Improvement (MEDIUM PRIORITY)
- **Objective**: Implement advanced training techniques
- **Tasks**:
  - Research and implement:
    1. Label Smoothing
    2. Class imbalance handling in loss function (Loss Weighting)
    3. Gradient accumulation and Clipping optimization
    4. Scheduler improvements
    5. Check padding alignment
    6. No EOS token optimization
- **Rationale**: Advanced techniques for incremental performance improvements

### Phase 4: Advanced Features (Week 5)

#### 8. NER-7: Add HyperParameter optimization Strategy (MEDIUM PRIORITY)
- **Objective**: Automated hyperparameter tuning
- **Tasks**:
  - Implement Bayesian Optimization with Optuna
  - Add Hyperband with Ray Tune option
  - Implement Random Search with HF Trainer
- **Rationale**: Scales experimentation, requires stable foundation

#### 9. NER-4: Add new train sets (LOW PRIORITY)
- **Objective**: Expand training data diversity
- **Tasks**:
  - Research for new training datasets
  - Add new datasets to training process
  - Keep validation separate for every dataset
- **Rationale**: Dataset diversity, best done after optimizations

## Success Metrics
- Code quality: 0 linting errors, 100% type coverage
- Test coverage: 85%+ for all modules
- Model performance: F1 score > 0.8
- Security: No critical vulnerabilities
- Infrastructure: Seamless offline/online training

## Timeline
- **Week 1**: Complete Phase 1 (Code Quality Foundation)
- **Week 2**: Complete Phase 2 (Infrastructure & Security)
- **Weeks 3-4**: Complete Phase 3 (Model Performance)
- **Week 5**: Complete Phase 4 (Advanced Features)

## Dependencies
- Phase 1 must be completed before other phases to avoid technical debt
- Phase 3 depends on stable test infrastructure from Phase 1
- Phase 4 depends on optimizations from Phase 3