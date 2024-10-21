const { parentPort, workerData } = require('worker_threads');
const { analyzeCode } = require('../services/codeAnalyzer.js');

async function runAnalysis() {
  try {
    const { files, projectId } = workerData;
    const analysisResults = await analyzeCode(files);
    parentPort.postMessage({ projectId, analysisResults });
  } catch (error) {
    parentPort.postMessage({ error: error.message, projectId: workerData.projectId });
  }
}

runAnalysis();