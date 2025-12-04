# Test Suite

Comprehensive test suite for Travel Agency RAG System with 100% code coverage.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for component interactions
├── e2e/              # End-to-end tests for full system
├── conftest.py       # Shared fixtures and configuration
└── test_all.py       # Test runner script
```

## Running Tests

### Run all tests with coverage
```bash
pytest tests/ -v --cov=retrieval --cov=indexing --cov=app --cov-report=html
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# End-to-end tests only
pytest tests/e2e/ -v
```

### Run with coverage report
```bash
pytest tests/ --cov --cov-report=term-missing --cov-report=html
```

## Test Coverage

Target: **100% code coverage**

Coverage includes:
- All retrieval modules
- All indexing modules
- All API endpoints
- Error handling
- Edge cases

## Test Categories

### Unit Tests
- ActivityMatcher
- QueryRewriter
- EmbeddingGenerator
- BaselineRetriever
- ImprovedRetriever
- VectorRetriever
- QdrantStore
- LLMExtractor

### Integration Tests
- Indexing pipeline
- Retrieval systems
- Hybrid retrieval
- Component interactions

### End-to-End Tests
- API endpoints
- Full search workflows
- Performance requirements
- Error scenarios

## Performance Requirements

Tests verify:
- Query response time < 1 second (excluding LLM calls)
- Activity matching < 10ms per operation
- System handles 1000+ documents
- Memory usage < 500MB for 1000 documents

