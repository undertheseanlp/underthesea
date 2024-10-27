import React from 'react';
import { Link } from 'react-router-dom';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { APP_NAME } from './Config';

const Nav: React.FC = () => {
  // const handleGoogleLoginSuccess = (response: any) => {
  //   console.log('Google login successful:', response);
  //   // Handle the Google login response here, e.g., send the token to your backend for verification.
  // };

  // const handleGoogleLoginError = () => {
  //   console.log('Google login failed');
  // };

  return (
    <GoogleOAuthProvider clientId="1019763033940-c42boad0dt819pddir1g4jelvarm8802.apps.googleusercontent.com">
      <nav className="w-full p-4 bg-white shadow-md fixed top-0 z-50">
        <ul className="flex justify-start space-x-8">
          <li>
            <Link
              to="/"
              className="text-blue-600 font-semibold hover:text-blue-800 transition duration-300 ease-in-out px-4 py-2 rounded-md hover:bg-blue-100"
            >
              {APP_NAME}
            </Link>
          </li>
          {/* <li>
            <GoogleLogin
              onSuccess={handleGoogleLoginSuccess}
              onError={handleGoogleLoginError}
            />
          </li> */}
        </ul>
      </nav>
    </GoogleOAuthProvider>
  );
};

export default Nav;