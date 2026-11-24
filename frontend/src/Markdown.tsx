import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Renders model output as Markdown. Raw HTML is never rendered (react-markdown escapes it). */
export default function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer noopener" />,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
