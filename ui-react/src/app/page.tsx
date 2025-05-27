'use client';

import { useState } from 'react';

export default function Home() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    setLoading(true);
    setAnswer('');
    try {
      const res = await fetch('http://localhost:5050/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      setAnswer(data.answer?.result || 'No response');
    } catch (err) {
      console.error(err);
      setAnswer('Error fetching answer');
    }
    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center p-8">
      <h1 className="text-3xl font-bold mb-6">Ask the RAG System</h1>
      <input
        type="text"
        placeholder="Type your question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        className="p-3 w-full max-w-xl rounded bg-gray-800 text-white mb-4 border border-gray-700"
      />
      <button
        onClick={handleAsk}
        disabled={loading || !question}
        className="px-6 py-2 bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? 'Asking...' : 'Ask'}
      </button>

      {answer && (
        <div className="mt-6 max-w-2xl p-4 bg-gray-800 rounded border border-gray-700 whitespace-pre-wrap">
          {answer}
        </div>
      )}
    </main>
  );
}
