export default function Footer() {
  return (
    <footer className="bg-gray-900/50 border-t border-gray-800 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center space-y-6 md:space-y-0">
          {/* Project Info - Left Side */}
          <div className="text-left">
            <h3 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
              Drishti
            </h3>
            <p className="text-gray-400">
              Real-Time Crowd Understanding for Intelligent Safety
            </p>
          </div>

          {/* Contact Us - Right Side */}
          <div className="text-left md:text-right">
            <h4 className="text-lg font-semibold text-white mb-3">
              Contact Us
            </h4>
            <div className="space-y-2">
              <div className="flex items-center space-x-2 md:justify-end">
                <svg
                  className="w-5 h-5 text-blue-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
                <a
                  href="mailto:gondkaromkar53@gmail.com"
                  className="text-gray-400 hover:text-blue-400 transition-colors duration-200"
                >
                  gondkaromkar53@gmail.com
                </a>
              </div>
              <div className="flex items-center space-x-2 md:justify-end">
                <svg
                  className="w-5 h-5 text-blue-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
                  />
                </svg>
                <a
                  href="tel:+918855916700"
                  className="text-gray-400 hover:text-blue-400 transition-colors duration-200"
                >
                  +91 88559 16700
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
