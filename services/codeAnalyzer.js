const fs = require('fs/promises');
const path = require('path');
const { parse } = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const { ESLint } = require('eslint');
const { HTMLHint } = require('htmlhint');

let eslintConfig, htmlhintConfig, stylelintConfig, eslint, stylelint;

async function initializeConfigs() {
  try {
    eslintConfig = {
      env: {
        browser: true,
        es2021: true,
        node: true
      },
      extends: ['eslint:recommended'],
      parserOptions: {
        ecmaVersion: 12,
        sourceType: 'module'
      },
      rules: {}
    };
    htmlhintConfig = {};
    stylelintConfig = {};

    eslint = new ESLint({
      useEslintrc: false,
      baseConfig: eslintConfig,
      fix: true,
    });

    const stylelintModule = await import('stylelint');
    stylelint = stylelintModule.default;
  } catch (error) {
    console.error('Error initializing configs:', error);
    throw error;
  }
}

async function analyzeCode(files) {
  await initializeConfigs();
  const results = [];

  for (const file of files) {
    console.log(`Analyzing file: ${file.name}`);
    const fileContent = await fs.readFile(file.path, 'utf8');
    const fileExtension = path.extname(file.name).toLowerCase();
    const analysisResult = {
      fileName: file.name,
      issuesFound: [],
      codeSmells: [],
      metrics: { complexity: 0 },
      suggestions: []
    };

    switch (fileExtension) {
      case '.html':
        analyzeHTML(fileContent, analysisResult);
        break;
      case '.css':
        await analyzeCSS(fileContent, analysisResult);
        break;
      case '.js':
      case '.jsx':
      case '.ts':
      case '.tsx':
        await analyzeJavaScript(fileContent, file.path, analysisResult);
        break;
      default:
        analysisResult.issuesFound.push({
          message: `Unsupported file type: ${fileExtension}`,
          line: 1,
          column: 1,
        });
    }

    console.log(`Analysis result for ${file.name}:`, JSON.stringify(analysisResult, null, 2));
    results.push(analysisResult);
  }

  return results;
}

function analyzeHTML(content, result) {
  const messages = HTMLHint.verify(content, htmlhintConfig);
  messages.forEach(message => {
    result.issuesFound.push({
      message: message.message,
      line: message.line,
      column: message.col,
      ruleId: message.rule.id
    });
    result.suggestions.push({
      message: message.message,
      line: message.line,
      column: message.col,
      suggestion: message.rule.description
    });
  });
}

async function analyzeCSS(content, result) {
  try {
    const stylelintResult = await stylelint.lint({
      code: content,
      config: stylelintConfig,
      fix: true
    });

    stylelintResult.results[0].warnings.forEach(warning => {
      result.issuesFound.push({
        message: warning.text,
        line: warning.line,
        column: warning.column,
        ruleId: warning.rule
      });
      result.suggestions.push({
        message: warning.text,
        line: warning.line,
        column: warning.column,
        suggestion: stylelintResult.results[0].code ? 'Auto-fixed. Please check the updated file.' : warning.text
      });
    });
  } catch (error) {
    result.issuesFound.push({
      message: `CSS analysis error: ${error.message}`,
      line: 1,
      column: 1,
    });
  }
}

async function analyzeJavaScript(content, filePath, result) {
  try {
    const ast = parse(content, {
      sourceType: 'module',
      plugins: ['jsx', 'typescript'],
    });

    let complexity = 1;
    traverse(ast, {
      FunctionDeclaration() { complexity++; },
      FunctionExpression() { complexity++; },
      ArrowFunctionExpression() { complexity++; },
      IfStatement() { complexity++; },
      ForStatement() { complexity++; },
      WhileStatement() { complexity++; },
      DoWhileStatement() { complexity++; },
      SwitchCase() { complexity++; },
    });
    result.metrics.complexity = complexity;

    detectCodeSmells(ast, result);

    const eslintResults = await eslint.lintText(content, { filePath });
    for (const eslintResult of eslintResults) {
      for (const message of eslintResult.messages) {
        result.issuesFound.push({
          message: message.message,
          line: message.line,
          column: message.column,
          ruleId: message.ruleId,
        });
        result.suggestions.push({
          message: message.message,
          line: message.line,
          column: message.column,
          suggestion: message.fix ? 'Auto-fixed. Please check the updated file.' : (message.suggestions && message.suggestions.length > 0 ? message.suggestions[0].desc : message.message)
        });
      }
    }
  } catch (error) {
    result.issuesFound.push({
      message: `Analysis error: ${error.message}`,
      line: error.loc ? error.loc.line : 1,
      column: error.loc ? error.loc.column : 1,
    });
  }
}

function detectCodeSmells(ast, analysisResult) {
  const longFunctionThreshold = 50;
  const highComplexityThreshold = 10;

  traverse(ast, {
    Function(path) {
      const start = path.node.loc.start.line;
      const end = path.node.loc.end.line;
      const functionLength = end - start + 1;

      if (functionLength > longFunctionThreshold) {
        analysisResult.codeSmells.push({
          type: 'Long Function',
          message: `Function is too long (${functionLength} lines)`,
          line: start,
        });
        analysisResult.suggestions.push({
          message: `Function is too long (${functionLength} lines)`,
          line: start,
          column: 1,
          suggestion: 'Consider breaking this function into smaller, more manageable functions.'
        });
      }

      let complexity = 1;
      path.traverse({
        IfStatement() { complexity++; },
        ForStatement() { complexity++; },
        WhileStatement() { complexity++; },
        DoWhileStatement() { complexity++; },
        SwitchCase() { complexity++; },
      });

      if (complexity > highComplexityThreshold) {
        analysisResult.codeSmells.push({
          type: 'High Complexity',
          message: `Function has high complexity (${complexity})`,
          line: start,
        });
        analysisResult.suggestions.push({
          message: `Function has high complexity (${complexity})`,
          line: start,
          column: 1,
          suggestion: 'Consider refactoring this function to reduce its complexity. Break it down into smaller functions or simplify the logic.'
        });
      }
    },
  });
}

module.exports = { analyzeCode };


/* const fs = require('fs/promises');
const path = require('path');
const { parse } = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const { ESLint } = require('eslint');
const { HTMLHint } = require('htmlhint');

let eslintConfig, htmlhintConfig, stylelintConfig, eslint, stylelint;

async function initializeConfigs() {
  eslintConfig = require(path.resolve(process.cwd(), '.eslintrc.js'));
  htmlhintConfig = JSON.parse(await fs.readFile(path.resolve(process.cwd(), '.htmlhintrc'), 'utf8'));
  stylelintConfig = JSON.parse(await fs.readFile(path.resolve(process.cwd(), '.stylelintrc.json'), 'utf8'));

  eslint = new ESLint({
    overrideConfigFile: true,
    baseConfig: eslintConfig,
    fix: true,
  });

  const stylelintModule = await import('stylelint');
  stylelint = stylelintModule.default;
}

async function analyzeCode(files) {
  await initializeConfigs();
  const results = [];

  for (const file of files) {
    console.log(`Analyzing file: ${file.name}`);
    const fileContent = await fs.readFile(file.path, 'utf8');
    const fileExtension = path.extname(file.name).toLowerCase();
    const analysisResult = {
      fileName: file.name,
      issuesFound: [],
      codeSmells: [],
      metrics: { complexity: 0 },
      suggestions: []
    };

    switch (fileExtension) {
      case '.html':
        analyzeHTML(fileContent, analysisResult);
        break;
      case '.css':
        await analyzeCSS(fileContent, analysisResult);
        break;
      case '.js':
      case '.jsx':
      case '.ts':
      case '.tsx':
        await analyzeJavaScript(fileContent, file.path, analysisResult);
        break;
      default:
        analysisResult.issuesFound.push({
          message: `Unsupported file type: ${fileExtension}`,
          line: 1,
          column: 1,
        });
    }

    console.log(`Analysis result for ${file.name}:`, JSON.stringify(analysisResult, null, 2));
    results.push(analysisResult);
  }

  return results;
}

function analyzeHTML(content, result) {
  const messages = HTMLHint.verify(content, htmlhintConfig);
  messages.forEach(message => {
    result.issuesFound.push({
      message: message.message,
      line: message.line,
      column: message.col,
      ruleId: message.rule.id
    });
    result.suggestions.push({
      message: message.message,
      line: message.line,
      column: message.col,
      suggestion: message.rule.description
    });
  });
}

async function analyzeCSS(content, result) {
  try {
    const stylelintResult = await stylelint.lint({
      code: content,
      config: stylelintConfig,
      fix: true
    });

    stylelintResult.results[0].warnings.forEach(warning => {
      result.issuesFound.push({
        message: warning.text,
        line: warning.line,
        column: warning.column,
        ruleId: warning.rule
      });
      result.suggestions.push({
        message: warning.text,
        line: warning.line,
        column: warning.column,
        suggestion: stylelintResult.results[0].code ? 'Auto-fixed. Please check the updated file.' : warning.text
      });
    });
  } catch (error) {
    result.issuesFound.push({
      message: `CSS analysis error: ${error.message}`,
      line: 1,
      column: 1,
    });
  }
}

async function analyzeJavaScript(content, filePath, result) {
  try {
    const ast = parse(content, {
      sourceType: 'module',
      plugins: ['jsx', 'typescript'],
    });

    let complexity = 1;
    traverse(ast, {
      FunctionDeclaration() { complexity++; },
      FunctionExpression() { complexity++; },
      ArrowFunctionExpression() { complexity++; },
      IfStatement() { complexity++; },
      ForStatement() { complexity++; },
      WhileStatement() { complexity++; },
      DoWhileStatement() { complexity++; },
      SwitchCase() { complexity++; },
    });
    result.metrics.complexity = complexity;

    detectCodeSmells(ast, result);

    const eslintResults = await eslint.lintText(content, { filePath });
    for (const eslintResult of eslintResults) {
      for (const message of eslintResult.messages) {
        result.issuesFound.push({
          message: message.message,
          line: message.line,
          column: message.column,
          ruleId: message.ruleId,
        });
        result.suggestions.push({
          message: message.message,
          line: message.line,
          column: message.column,
          suggestion: message.fix ? 'Auto-fixed. Please check the updated file.' : (message.suggestions && message.suggestions.length > 0 ? message.suggestions[0].desc : message.message)
        });
      }
    }
  } catch (error) {
    result.issuesFound.push({
      message: `Analysis error: ${error.message}`,
      line: error.loc ? error.loc.line : 1,
      column: error.loc ? error.loc.column : 1,
    });
  }
}

function detectCodeSmells(ast, analysisResult) {
  const longFunctionThreshold = 50;
  const highComplexityThreshold = 10;

  traverse(ast, {
    Function(path) {
      const start = path.node.loc.start.line;
      const end = path.node.loc.end.line;
      const functionLength = end - start + 1;

      if (functionLength > longFunctionThreshold) {
        analysisResult.codeSmells.push({
          type: 'Long Function',
          message: `Function is too long (${functionLength} lines)`,
          line: start,
        });
        analysisResult.suggestions.push({
          message: `Function is too long (${functionLength} lines)`,
          line: start,
          column: 1,
          suggestion: 'Consider breaking this function into smaller, more manageable functions.'
        });
      }

      let complexity = 1;
      path.traverse({
        IfStatement() { complexity++; },
        ForStatement() { complexity++; },
        WhileStatement() { complexity++; },
        DoWhileStatement() { complexity++; },
        SwitchCase() { complexity++; },
      });

      if (complexity > highComplexityThreshold) {
        analysisResult.codeSmells.push({
          type: 'High Complexity',
          message: `Function has high complexity (${complexity})`,
          line: start,
        });
        analysisResult.suggestions.push({
          message: `Function has high complexity (${complexity})`,
          line: start,
          column: 1,
          suggestion: 'Consider refactoring this function to reduce its complexity. Break it down into smaller functions or simplify the logic.'
        });
      }
    },
  });
}

module.exports = { analyzeCode };



 */