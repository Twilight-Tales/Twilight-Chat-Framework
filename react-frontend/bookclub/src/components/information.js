
function InformationComponent({ book, author, pages, currentChapter, currentPage, startDate, progress }) {
return (
    <div className="information-panel">
    <h2>Information</h2>
    <div className="book-details">
        <p><strong>Book:</strong> {book}</p>
        <p><strong>Author:</strong> {author}</p>
        <p><strong>Pages:</strong> {pages}</p>
    </div>
    <div className="current-status">
        <p><strong>Current Status:</strong></p>
        <p>Chapter {currentChapter} Page {currentPage}</p>
        <progress value={progress} max="100">{progress}%</progress>
        <p>{progress}% finished</p>
    </div>
    <div className="start-date">
        <p><strong>Date Started:</strong> {startDate}</p>
    </div>
    </div>
);
}

export default InformationComponent;