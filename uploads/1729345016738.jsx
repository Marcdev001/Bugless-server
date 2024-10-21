import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom'
import BackToTop from './BackToTop'; // Adjust the path as needed
import '../Testimonials.css'


const Testimonials = () => {
  const [showAll, setShowAll] = useState(false);
  const [userTestimonials, setUserTestimonials] = useState([]);
  const [newTestimonial, setNewTestimonial] = useState('');
  const [userName, setUserName] = useState('');

  // Load testimonials from local storage on component mount
  useEffect(() => {
    const storedTestimonials = JSON.parse(localStorage.getItem('userTestimonials'));
    if (storedTestimonials) {
      setUserTestimonials(storedTestimonials);
    }
  }, []); // Only runs once on mount

  const handleSubmit = () => {
    if (newTestimonial.trim() !== '' && userName.trim() !== '') {
      const testimonial = { name: userName, message: newTestimonial };
      const updatedTestimonials = [...userTestimonials, testimonial];
      setUserTestimonials(updatedTestimonials);
      setNewTestimonial('');
      setUserName('');
      // Save updated testimonials to local storage
      localStorage.setItem('userTestimonials', JSON.stringify(updatedTestimonials));
    }
  };

  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div className={`testimonials-container ${isVisible ? 'fade-in' : ''}`}>
      <h1>User Testimonials</h1>
      <ul>
        {userTestimonials.slice(0, showAll ? userTestimonials.length : 3).map((testimonial, index) => (
          <li key={index} className='testimonials-card'>
            <i className='fa solid fa-user testimonials-icon'></i> <div className='name'>{testimonial.name}:</div> {testimonial.message}
          </li>
        ))}
      </ul>
      {!showAll && (
        <button onClick={() => setShowAll(true)}>See All Testimonials</button>
      )}
      <div className='form-container'>
        <input
          type="text"
          value={userName}
          onChange={(e) => setUserName(e.target.value)}
          placeholder="Your Name"
        />
        <textarea
          value={newTestimonial}
          onChange={(e) => setNewTestimonial(e.target.value)}
          placeholder="Write your testimonial..."
        />
        <button onClick={handleSubmit} className='btn-submit'>Send</button>
      </div>




      <BackToTop />
    </div>
  );
};

export default Testimonials;

