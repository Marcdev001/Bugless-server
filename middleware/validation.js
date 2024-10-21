const mongoose = require('mongoose');

function validateProjectId(req, res, next) {
  const projectId = req.params.projectId || req.body.projectId;
  if (projectId && !mongoose.Types.ObjectId.isValid(projectId)) {
    return res.status(400).json({ error: 'Invalid project ID' });
  }
  next();
}

module.exports = { validateProjectId };