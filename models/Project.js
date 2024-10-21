const mongoose = require('mongoose');

const ProjectSchema = new mongoose.Schema({
  files: [{
    name: String,
    path: String
  }],
  analysisResults: [{
    fileName: String,
    issuesFound: [{
      message: String,
      line: Number,
      column: Number,
      ruleId: String
    }],
    codeSmells: [{
      type: String,
      message: String,
      line: Number
    }],
    metrics: {
      complexity: Number
    }
  }],
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Project', ProjectSchema);