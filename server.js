const express = require('express');
const mongoose = require('mongoose');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs/promises');
const helmet = require('helmet');
const dotenv = require('dotenv');
const { errorHandler } = require('./middleware/errorHandler.js');
const Project = require('./models/Project.js');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5003;

app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' }
}));

// Update CORS configuration
app.use(cors({
  origin: [
    'https://bugless-application.vercel.app',
    'http://localhost:3000',
    'http://localhost:5173', // Add Vite development server
    'http://127.0.0.1:5173'  // Also add this variant
  ],
  credentials: true,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  maxAge: 86400 // Increase preflight cache to 24 hours
}));

app.use(require('compression')());

app.use(express.json());

const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedExtensions = ['.html', '.css', '.js', '.jsx', '.ts', '.tsx', '.py'];
    const fileExtension = path.extname(file.originalname).toLowerCase();
    if (allowedExtensions.includes(fileExtension)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only supported code files are allowed.'));
    }
  },
  limits: {
    fileSize: 5 * 1024 * 1024
  }
});

app.post('/api/analyze', upload.array('files', 20), async (req, res, next) => {
  const startTime = Date.now();
  
  try {
    req.setTimeout(30000);
    
    const files = req.files.map(file => ({
      name: file.originalname,
      path: file.path
    }));

    const { analyzeCode } = require('./services/codeAnalyzer');
    const analysisResults = await analyzeCode(files);

    const processingTime = Date.now() - startTime;
    res.set('X-Response-Time', `${processingTime}ms`);

    res.json({ 
      analysisResults,
      performance: {
        processingTime,
        filesAnalyzed: files.length
      }
    });
  } catch (error) {
    console.error('Error in /api/analyze:', error);
    next(error);
  } finally {
    if (req.files) {
      for (const file of req.files) {
        fs.unlink(file.path).catch(err => console.error('Error deleting file:', err));
      }
    }
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.use(errorHandler);

mongoose.connect(process.env.MONGODB_URI)
  .then(() => {
    console.log('Connected to MongoDB');
    const startServer = (port) => {
      const server = app.listen(port)
        .once('listening', () => {
          console.log(`Server is running on port ${port}`);
        })
        .once('error', (err) => {
          if (err.code === 'EADDRINUSE') {
            console.log(`Port ${port} is busy, trying ${port + 1}...`);
            server.close();
            startServer(port + 1);
          } else {
            console.error('Server error:', err);
          }
        });
    };

    startServer(PORT);
  })
  .catch((err) => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

mongoose.connection.on('error', err => {
  console.error('MongoDB connection error:', err);
});