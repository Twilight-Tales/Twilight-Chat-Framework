import { useEffect } from "react";

import React, { useState } from 'react';

import { ChainlitAPI, sessionState, useChatSession } from "@chainlit/react-client";
import { useRecoilValue } from "recoil";
import './App.css';
import ChatbotComponent from './components/chat';
import InformationComponent from './components/information';

const CHAINLIT_SERVER = "http://localhost:9999";

const apiClient = new ChainlitAPI(CHAINLIT_SERVER);

function App() {
  const [isInfoVisible, setInfoVisible] = useState(false);
  const [bookDetails, setBookDetails] = useState({
    book: 'Example Book Title',
    author: 'Author Name',
    pages: 300,
    currentChapter: 5,
    currentPage: 150,
    startDate: '01/01/2024',
    progress: 50
  });
  const toggleInfo = () => setInfoVisible(!isInfoVisible);

  const { connect } = useChatSession();
  const session = useRecoilValue(sessionState);

  useEffect(() => {
    if (session?.socket.connected) {
      return
    }
    connect({
      client: apiClient
    });
    
  }, [session, connect]);


  return (
      <div className="App">
        <table>
          <tr>
            <td>
              {isInfoVisible && <InformationComponent {...bookDetails} />}
            </td>
          </tr>
          <tr>
            <td>
              <div className="chatbot">
                <ChatbotComponent />
                <button onClick={toggleInfo} className="info-toggle">
                  i
                </button>
              </div>
            </td>
          </tr>
        </table>
      </div>
    );
}

export default App;
