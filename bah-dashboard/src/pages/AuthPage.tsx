import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import GlassSurface from '@/components/ui/GlassSurface';
import SatelliteDishIcon from '@/components/ui/SatelliteDishIcon';

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isLogin) {
        // Login uses OAuth2 format (FormData)
        const formData = new URLSearchParams();
        formData.append('username', email);
        formData.append('password', password);

        const response = await fetch('http://127.0.0.1:8000/auth/login', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          login(data.access_token);
          navigate('/dashboard');
        } else {
          const errData = await response.json();
          setError(errData.detail || 'Login failed');
        }
      } else {
        // Signup uses JSON
        const response = await fetch('http://127.0.0.1:8000/auth/signup', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ email, password }),
        });

        if (response.ok) {
          // Auto login after signup
          const formData = new URLSearchParams();
          formData.append('username', email);
          formData.append('password', password);

          const loginResponse = await fetch('http://127.0.0.1:8000/auth/login', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData,
          });

          if (loginResponse.ok) {
            const data = await loginResponse.json();
            login(data.access_token);
            navigate('/dashboard');
          }
        } else {
          const errData = await response.json();
          setError(errData.detail || 'Signup failed');
        }
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', padding: '2rem' }}>
      <GlassSurface width={400} borderRadius={24} style={{ padding: '3rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem' }}>
          
          <div style={{ 
            width: 48, height: 48, 
            borderRadius: 12, 
            background: `linear-gradient(135deg, rgba(164, 173, 181, 0.15), rgba(164, 173, 181, 0.05))`,
            border: `1px solid rgba(255, 255, 255, 0.1)`,
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: `0 0 16px rgba(255, 107, 53, 0.15)`
          }}>
            <SatelliteDishIcon size={24} color="#FF6B35" />
          </div>

          <div style={{ textAlign: 'center' }}>
            <h1 style={{ 
              fontSize: 24, fontWeight: 700, margin: 0,
              background: `linear-gradient(90deg, #FFFFFF, #9A9A9E)`,
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              fontFamily: "'Space Grotesk', sans-serif" 
            }}>
              {isLogin ? 'Welcome Back' : 'Initialize Access'}
            </h1>
            <p style={{ color: '#9A9A9E', fontSize: 14, marginTop: '8px' }}>
              {isLogin ? 'Enter your credentials to access the dashboard' : 'Create an account to begin processing'}
            </p>
          </div>

          {error && (
            <div style={{
              background: 'rgba(255, 79, 94, 0.1)',
              border: '1px solid rgba(255, 79, 94, 0.3)',
              color: '#FF4F5E',
              padding: '10px 16px',
              borderRadius: '8px',
              fontSize: 14,
              width: '100%',
              textAlign: 'center'
            }}>
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={{
                width: '100%',
                padding: '12px 16px',
                background: 'rgba(20, 20, 22, 0.6)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '12px',
                color: '#F5F5F5',
                outline: 'none',
                transition: 'all 0.2s',
                fontFamily: "'Space Grotesk', sans-serif",
                fontSize: 14
              }}
              onFocus={(e) => e.target.style.borderColor = 'rgba(255, 107, 53, 0.5)'}
              onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.08)'}
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={{
                width: '100%',
                padding: '12px 16px',
                background: 'rgba(20, 20, 22, 0.6)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '12px',
                color: '#F5F5F5',
                outline: 'none',
                transition: 'all 0.2s',
                fontFamily: "'Space Grotesk', sans-serif",
                fontSize: 14
              }}
              onFocus={(e) => e.target.style.borderColor = 'rgba(255, 107, 53, 0.5)'}
              onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.08)'}
            />

            <button
              type="submit"
              disabled={loading}
              style={{
                marginTop: '1rem',
                width: '100%',
                padding: '14px',
                background: loading ? 'rgba(255,255,255,0.05)' : '#FF6B35',
                border: 'none',
                borderRadius: '12px',
                color: loading ? '#55555A' : '#0A0A0B',
                fontWeight: 700,
                fontFamily: "'Space Grotesk', sans-serif",
                fontSize: 14,
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => { if(!loading) e.currentTarget.style.filter = 'brightness(1.1)'; }}
              onMouseLeave={(e) => { if(!loading) e.currentTarget.style.filter = 'brightness(1)'; }}
            >
              {loading ? 'Processing...' : isLogin ? 'Authenticate' : 'Establish Link'}
            </button>
          </form>

          <div style={{ color: '#55555A', fontSize: 14 }}>
            {isLogin ? "Don't have clearance? " : "Already initialized? "}
            <span 
              onClick={() => setIsLogin(!isLogin)}
              style={{ color: '#FF6B35', cursor: 'pointer', fontWeight: 500 }}
            >
              {isLogin ? 'Request Access' : 'Login'}
            </span>
          </div>

        </div>
      </GlassSurface>
    </div>
  );
}
