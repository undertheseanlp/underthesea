import React from 'react';
import { Link } from 'react-router-dom';

const Nav: React.FC = () => {
  return (
    <nav className="w-full p-4 bg-white shadow-md fixed top-0 z-50">
      <ul className="flex justify-start space-x-8">
        <li>
          <Link
            to="/"
            className="text-blue-600 font-semibold hover:text-blue-800 transition duration-300 ease-in-out px-4 py-2 rounded-md hover:bg-blue-100"
          >
            Languages
          </Link>
        </li>
      </ul>
    </nav>
  );
};

export default Nav;