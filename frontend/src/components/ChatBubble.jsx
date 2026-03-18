import ReactMarkdown from "react-markdown";

const styles = {
  aiRow: {
    display: "flex",
    flexDirection: "column",
    animation: "fadeIn 0.3s ease",
  },
  aiDivider: {
    height: 1,
    background: "var(--border)",
    marginBottom: 24,
  },
  aiContent: {
    maxWidth: 720,
    lineHeight: 1.8,
    fontSize: 15,
    color: "var(--text)",
  },
  userRow: {
    display: "flex",
    justifyContent: "flex-end",
    animation: "fadeIn 0.3s ease",
  },
  userBubble: {
    maxWidth: "70%",
    padding: "10px 16px",
    borderRadius: 18,
    borderTopRightRadius: 4,
    background: "var(--accent)",
    color: "#fff",
    fontSize: 15,
    lineHeight: 1.7,
    whiteSpace: "pre-wrap",
  },
};

export default function ChatBubble({ role, content }) {
  if (role === "user") {
    return (
      <div style={styles.userRow}>
        <div style={styles.userBubble}>{content}</div>
      </div>
    );
  }

  return (
    <div style={styles.aiRow}>
      <div style={styles.aiDivider} />
      <div style={styles.aiContent}>
        <div className="md-content">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
