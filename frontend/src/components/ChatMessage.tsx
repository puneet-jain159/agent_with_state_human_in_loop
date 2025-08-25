import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck, faChevronDown } from '@fortawesome/free-solid-svg-icons';
import copyIconUrl from '../assets/images/copy_icon.svg';
import buttonIconUrl from '../assets/images/buttonIcon.svg';
import downIconUrl from '../assets/images/down_icon.svg';
import thumbsUpIconUrl from '../assets/images/thumbs_up_icon.svg';
import thumbsDownIconUrl from '../assets/images/thumbs_down_icon.svg';
import { Message } from '../types';
import { useChat } from '../context/ChatContext';
import sourceIconUrl from '../assets/images/source_icon.svg';
import remarkGfm from 'remark-gfm';
import { submitFeedback } from '../api/chatApi';
import FeedbackPopup from './FeedbackPopup';
import { buildTraceUrl, fetchBackendConfig, BackendConfig } from '../config';

const MessageContainer = styled.div<{ isUser: boolean }>`
  display: flex;
  flex-direction: column;
  width: 100%;
  align-items: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  align-self: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  max-width: ${props => props.isUser ? '80%' : '100%'};
  margin-bottom: ${props => props.isUser ? '10px' : '0px'};
  color: #333333;
`;

const UserMessageContent = styled.div`
  background-color: #F5F5F5;
  color: #11171C;
  padding: 8px 16px;
  border-radius: 12px;
  font-size: 15px;
  line-height: 1.5;
  word-wrap: break-word;
  overflow-wrap: break-word;
  white-space: normal;
  > p {
    margin: 0px;
  }
`;

const BotMessageContent = styled.div`
  border-radius: 12px;
  width: 100%;
  padding: 6px;
  word-wrap: break-word;
  overflow-wrap: break-word;
  white-space: normal;
  text-align: left;
  font-size: 15px;
  margin: 2px 0;
`;

const ModelInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
`;

const ModelIcon = styled.div`
  width: 20px;
  height: 20px;
  background-image: url(${buttonIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
`;

const ModelName = styled.span`
  font-size: 15px;
  color: #11171C;
  font-weight: 600;
`;

const ModelMetrics = styled.span`
  font-size: 11px;
  color: #5F7281;
`;

const MessageFooter = styled.div`
  display: flex;
  flex-direction: column;
  margin-top: 8px;
  gap: 8px;
  align-items: flex-start;
`;

const MessageActions = styled.div`
  display: flex;
  gap: 8px;
  margin-top: 4px;
  margin-bottom: 24px;
`;

const ActionButton = styled.button`
  width: 24px;
  height: 24px;
  border: none;
  background-color: transparent;
  cursor: pointer;
  display: flex;
  justify-content: center;
  align-items: center;
  
  &:hover {
    background-color: #F0F0F0;
    border-radius: 4px;
  }
`;

const CopyButton = styled(ActionButton)<{ copied: boolean }>`
  background-image: url(${props => props.copied ? '' : copyIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
    color: #0E538B;
  }
`;

const CheckIconWrapper = styled.div<{ $copied: boolean }>`
  display: none;
  color: #5F7281;
  font-size: 15px;
  ${props => props.$copied && `
    display: block;
  `}
`;

const ThumbsUpButton = styled(ActionButton)<{ isActive: boolean }>`
  background-image: url(${thumbsUpIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
    color: #0E538B;
  }
  ${props => props.isActive && `
    background-color: rgba(34, 114, 180, 0.08);
    color: #0E538B;
  `}
`;

const ThumbsDownButton = styled(ActionButton)<{ isActive: boolean }>`
  background-image: url(${thumbsDownIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
    color: #0E538B;
  }
  ${props => props.isActive && `
    background-color: rgba(34, 114, 180, 0.08);
    color: #0E538B;
  `}
`;

const SourcesSection = styled.div`
  margin-top: 16px;
  width: 100%;
`;

const SourceContent = styled.div`
  width: 100%;
  padding: 32px;
  background: #F5F5F5;
  box-shadow: 0px 1px 0px rgba(0, 0, 0, 0.02);
  border-radius: 8px;
  outline: 1px #D1D9E1 solid;
  outline-offset: -1px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 8px;
`;

const SourceItem = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const SourceIconContainer = styled.div`
  height: 32px;
  min-width: 32px;
  background: rgba(0, 0, 59, 0.05);
  border-radius: 4px;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const SourceIcon = styled.div`
  background-image: url(${sourceIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  width: 16px;
  height: 16px;
`;

const SourceTextContent = styled.div`
  width: 100%;
  height: 100%;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  display: flex;
`;

const SourceText = styled.div`
  color: #11171C;
  font-size: 12px;
  line-height: 1.5;
  width: 100%;
`;

const SourceMetadata = styled.div`
  color: #5F7281;
  font-size: 11px;
  line-height: 1.4;
  width: 100%;
`;

const SourcesButton = styled.button`
  background: none;
  border: none;
  color: #11171C;
  font-size: 15px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  border: 1px solid #E0E0E0;
  border-radius: 4px;
  padding-right: 24px;
  position: relative;
  
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
    border: 1px solid #2272B4;
    color: #0E538B;
  }

  &::after {
    content: "";
    position: absolute;
    right: 4px;
    width: 14px;
    height: 14px;
    background-image: url(${downIconUrl});
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
  }
`;

const SourceCardsContainer = styled.div`
  width: 100%;
  overflow-x: auto;
  display: flex;
  gap: 8px;
  margin-top: 8px;
  padding-bottom: 4px;
  
  /* Show scrollbar and style it */
  scrollbar-width: thin;
  scrollbar-color: #C0CDD8 #F5F5F5;
  
  /* Webkit scrollbar styles */
  &::-webkit-scrollbar {
    height: 4px;
    display: block;
  }

  &::-webkit-scrollbar-track {
    background: #F5F5F5;
    border-radius: 2px;
  }

  &::-webkit-scrollbar-thumb {
    background: #C0CDD8;
    border-radius: 2px;
    
    &:hover {
      background: #A0B0C0;
    }
  }
`;

const SourcePreviewCard = styled.div`
  min-width: 200px;
  max-width: 200px;
  padding: 16px;
  background: white;
  box-shadow: 0px 1px 0px rgba(0, 0, 0, 0.02);
  border-radius: 8px;
  outline: 1px #D1D9E1 solid;
  outline-offset: -1px;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 8px;
  display: flex;
  cursor: pointer;

  &:hover {
    outline: 1px #2272B4 solid;
    background: rgba(34, 114, 180, 0.08);
  }
`;

const SourcePreviewItem = styled.div`
  width: 100%;
  height: 100%;
  justify-content: flex-start;
  align-items: center;
  gap: 16px;
  display: flex;
`;

const PreviewText = styled.div`
  color: #11171C;
  font-size: 11px;
  line-height: 1.5;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  width: 100%;
`;

const ThinkingIndicator = styled.div`
  font-size: 15px;
  color: #5F7281;
  margin-bottom: 10px 0px;
  align-self: flex-start;
  text-align: left;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const Spinner = styled.div`
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 1px solid transparent;
  border-top: 1px solid #5F7281;
  border-right: 1px solid #5F7281;
  border-radius: 50%;
  animation: spin 0.5s linear infinite;
  margin-right: 8px;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ThinkContainer = styled.div`
  margin: 8px -6px;
  border: 1px solid #E0E0E0;
  border-radius: 8px;
  overflow: hidden;
`;

const ThinkHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px;
  border-bottom: 1px solid #e3e8ee;
  cursor: pointer;
  user-select: none;
  min-height: 40px;
  &:hover {
    background-color: #f6f9fc;
  }
`;

const ThinkTitle = styled.span`
  width: 100%;
  justify-content: space-between;
  color: #3b4252;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
`;

const ThinkContent = styled.div<{ isExpanded: boolean }>`
  padding: ${props => props.isExpanded ? '12px' : '0'};
  max-height: ${props => props.isExpanded ? '1000px' : '0'};
  overflow: hidden;
  background-color: white;
`;

const ChevronIcon = styled(FontAwesomeIcon)<{ isExpanded: boolean }>`
  color: #5F7281;
  margin-left: 8px;
  transform: rotate(${props => props.isExpanded ? '180deg' : '0deg'});
`;

const StyledLink = styled.a`
  color: #0066cc;
  text-decoration: none;
  &:hover {
    color: #004499;
    text-decoration: underline;
  }
`;

const TraceLink = styled.a`
  background-color: #E0E0E0;
  color: #11171C;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 4px;
  margin-left: 10px;
  text-decoration: none;
  &:hover {
    background-color: #D0D0D0;
    color: #0E538B;
  }
`;

interface ChatMessageProps {
  message: Message;
  'data-testid'?: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const { copyMessage, currentSessionId } = useChat();
  const isUser = message.role === 'user';
  const [showSources, setShowSources] = useState(false);
  const [selectedSource, setSelectedSource] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const [closedThinks, setClosedThinks] = useState<string[]>([]);
  const [rating, setRating] = useState<'up' | 'down' | null>(message.rating || null);
  const [feedbackPopup, setFeedbackPopup] = useState(false);
  const [backendConfig, setBackendConfig] = useState<BackendConfig | null>(null);
  const chatContentRef = useRef<HTMLDivElement>(null);

  // Fetch backend configuration on component mount
  useEffect(() => {
    const loadConfig = async () => {
      const config = await fetchBackendConfig();
      if (config) {
        setBackendConfig(config);
      }
    };
    loadConfig();
  }, []);

  const handleCopy = async () => {
    await copyMessage(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 5000); // Reset after 5 seconds
  };

  const submitFeedbackAndClose = async (feedback: { rating: 'up' | 'down'; comment?: string }) => {
    try {
      // If we don't have a trace_id, try to fetch the latest message data from the backend
      let traceId = message.trace_id;
      if (!traceId) {
        try {
          const response = await fetch(`/chat-api/chats`);
          if (response.ok) {
            const data = await response.json();
            const messageData = data.sessions?.flatMap((s: any) => s.messages)?.find((m: any) => m.message_id === message.message_id);
            if (messageData?.trace_id) {
              traceId = messageData.trace_id;
            }
          }
        } catch (e) {
          // Silently handle fetch errors
        }
      }
      
      const result = await submitFeedback({
        message_id: message.message_id,
        session_id: currentSessionId || '',
        rating: feedback.rating,
        comment: feedback.comment,
        trace_id: traceId
      });
      
      // Update local rating state
      setRating(feedback.rating);
      
      return result;
    } catch (error) {
      console.error('Error submitting feedback:', error);
      return { success: false, message: 'Failed to submit feedback' };
    }
  };

  const toggleThink = (thinkId: string) => {
    setClosedThinks(prev => {
      const newSet = [...prev];
      const index = newSet.indexOf(thinkId);
      if (index >= 0) {
        newSet.splice(index, 1);
      } else {
        newSet.push(thinkId);
      }
      return newSet;
    });
  };

  const renderThinkContent = (message: Message) => {
    // Robustly parse <think> sections, including unclosed tags
    const thinkRegex = /<think>([\s\S]*?)(<\/think>|$)/g;
    let lastIndex = 0;
    let match;
    let thinkIndex = 0;
    const elements = [];
    const content = message.content;

    while ((match = thinkRegex.exec(content)) !== null) {
      // Add any regular content before this <think>
      if (match.index > lastIndex) {
        elements.push(
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content.slice(lastIndex, match.index)}</ReactMarkdown>
        );
      }

      const thinkId = 'think-' + thinkIndex;
      const isExpanded = !closedThinks.includes(thinkId);
      elements.push(
        <ThinkContainer key={thinkId}>
          <ThinkHeader onClick={() => toggleThink(thinkId)}>
            <ThinkTitle>
              {isExpanded ? 'Thoughts' : 'View thoughts'}
              <ChevronIcon
                icon={faChevronDown}
                isExpanded={isExpanded}
              />
            </ThinkTitle>
          </ThinkHeader>
          <ThinkContent isExpanded={isExpanded}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{match[1]}</ReactMarkdown>
          </ThinkContent>
        </ThinkContainer>
      );
      lastIndex = thinkRegex.lastIndex;
      thinkIndex++;
    }

    // insert a divider between content and footnotes
    let processedContent = content;
    const footnoteDefRegex = /^\[\^([^\]]+)\]:/m;
    const footNoteMatch = processedContent.match(footnoteDefRegex);
    if (footNoteMatch) {
      const insertPos = footNoteMatch.index;
      processedContent = processedContent.slice(0, insertPos) + '\n\n---\n\n' + processedContent.slice(insertPos);
    }

    if (lastIndex < processedContent.length) {
      elements.push(
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({node, ...props}) => {
              const container = document.querySelector('#messages-container');
              // scroll for footnote links + open in new tab for full urls
              const href = props.href || '';
              const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
                if (href.startsWith('#')) {
                  e.preventDefault();
                  const target = container?.querySelector(href);
                  console.log('target', target);
                  if (target) {
                    // Only scroll if not already in view
                    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                  }
                }
              };
              return (
                <StyledLink
                  {...props}
                  onClick={handleClick}
                  target={'_blank'}
                  rel={'noopener noreferrer'}
                >
                  {isNaN(Number(props.children)) ? props.children : `[${props.children}]`}
                </StyledLink>
              );
            }
          }}
        >
          {processedContent.slice(lastIndex)}
        </ReactMarkdown>
      );
    }

    return elements;
  };

  const renderSources = () => {
    if (!message.sources?.length) return null;

    return (
      <SourcesSection data-testid="sources-section">
        {selectedSource !== null ? (
          <>
            <SourcesButton onClick={() => setSelectedSource(null)}>
              Back to Sources
            </SourcesButton>
            <SourceContent data-testid="source-content">
              <SourceItem data-testid="source-item">
                <SourceTextContent data-testid="source-text-content">
                  <SourceText data-testid="source-text">{message.sources[selectedSource].page_content}</SourceText>
                  {message.sources[selectedSource].metadata?.url && (
                    <SourceMetadata data-testid="source-metadata">{message.sources[selectedSource].metadata.url}</SourceMetadata>
                  )}
                </SourceTextContent>
              </SourceItem>
            </SourceContent>
          </>
        ) : (
          <>
            <SourcesButton onClick={() => setShowSources(!showSources)}>
              Sources
            </SourcesButton>
            <SourceCardsContainer data-testid="source-cards-container">
              {message.sources.map((source, index) => (
                <SourcePreviewCard key={index} onClick={() => setSelectedSource(index)} data-testid="source-preview-card">
                  <SourcePreviewItem data-testid="source-preview-item">
                    <SourceIconContainer data-testid="source-icon-container">
                      <SourceIcon />
                    </SourceIconContainer>
                    <PreviewText data-testid="preview-text">
                      {source.page_content}
                    </PreviewText>
                  </SourcePreviewItem>
                </SourcePreviewCard>
              ))}
            </SourceCardsContainer>
          </>
        )}
      </SourcesSection>
    );
  };

  if (isUser) {
    return (
      <MessageContainer isUser={true} data-testid="user-message-container">
        <UserMessageContent data-testid="user-message-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </UserMessageContent>
      </MessageContainer>
    );
  }

  if (message.isThinking) {
    return (
      <MessageContainer isUser={false} data-testid="bot-message-container" style={{ marginBottom: '20px' }}>
        <ModelInfo data-testid="model-info">
          <ModelIcon data-testid="model-icon" />
          <ModelName data-testid="model-name">{'Procurement Bot'}</ModelName>
        </ModelInfo>
        <BotMessageContent ref={chatContentRef}>
          <ThinkingIndicator>
            <Spinner />
            Thinking...
          </ThinkingIndicator>
        </BotMessageContent>
      </MessageContainer>
    );
  }
  
  return (
    <MessageContainer isUser={false} data-testid="bot-message-container">
      <ModelInfo data-testid="model-info">
        <ModelIcon data-testid="model-icon" />
        <ModelName data-testid="model-name">
          {'Procurement Bot'}
        </ModelName>
        {message.trace_id && (
          <TraceLink 
            href={buildTraceUrl(message.trace_id, backendConfig || undefined)}
            target="_blank"
            rel="noopener noreferrer"
            title="View trace in MLflow"
          >
            🔍 Trace
          </TraceLink>
        )}
      </ModelInfo>
      
      <BotMessageContent data-testid="bot-message-content">
        {renderThinkContent(message)}
        {message.metrics && (
          <ModelMetrics>
            {message.metrics.timeToFirstToken && `${message.metrics.timeToFirstToken.toFixed(2)}s to first token + `}
            {message.metrics.totalTime && `${message.metrics.totalTime.toFixed(2)}s`}
          </ModelMetrics>
        )}
        {renderSources()}
        <MessageFooter>
          <MessageActions data-testid="message-actions">
            <CopyButton 
              onClick={handleCopy} 
              title="Copy" 
              copied={copied}
              data-testid={`copy-button-${message.message_id}`}
            >
              <CheckIconWrapper $copied={copied}>
                <FontAwesomeIcon icon={faCheck} />
              </CheckIconWrapper>
            </CopyButton>
            {!isUser && (
              <>
                <ThumbsUpButton
                  onClick={() => setFeedbackPopup(true)}
                  title="Provide feedback"
                  isActive={rating === 'up'}
                  data-testid={`thumbs-up-${message.message_id}`}
                />
                <ThumbsDownButton
                  onClick={() => setFeedbackPopup(true)}
                  title="Provide feedback"
                  isActive={rating === 'down'}
                  data-testid={`thumbs-down-${message.message_id}`}
                />
              </>
            )}
          </MessageActions>
        </MessageFooter>
      </BotMessageContent>
      <FeedbackPopup
        isOpen={feedbackPopup}
        onClose={() => setFeedbackPopup(false)}
        messageId={message.message_id}
        sessionId={currentSessionId || ''}
        onSubmit={submitFeedbackAndClose}
      />
    </MessageContainer>
  );
};

export default ChatMessage; 