import { useState } from 'react'
import { ShieldCheck, ChevronRight, ChevronLeft } from 'lucide-react'

interface Question {
  id: string
  question: string
  options: { label: string; score: number }[]
}

const QUESTIONS: Question[] = [
  {
    id: 'age',
    question: '您的年龄段是？',
    options: [
      { label: '25岁以下', score: 5 },
      { label: '25-35岁', score: 4 },
      { label: '36-50岁', score: 3 },
      { label: '51-60岁', score: 2 },
      { label: '60岁以上', score: 1 },
    ],
  },
  {
    id: 'income',
    question: '您的年收入水平？',
    options: [
      { label: '20万以下', score: 1 },
      { label: '20-50万', score: 2 },
      { label: '50-100万', score: 3 },
      { label: '100-300万', score: 4 },
      { label: '300万以上', score: 5 },
    ],
  },
  {
    id: 'experience',
    question: '您的投资经验如何？',
    options: [
      { label: '无投资经验', score: 1 },
      { label: '1-3年', score: 2 },
      { label: '3-5年', score: 3 },
      { label: '5-10年', score: 4 },
      { label: '10年以上', score: 5 },
    ],
  },
  {
    id: 'goal',
    question: '您的投资目标是什么？',
    options: [
      { label: '保值为主，略高于通胀', score: 1 },
      { label: '稳健增长，跑赢大盘', score: 3 },
      { label: '追求较高回报，可承受波动', score: 4 },
      { label: '追求最大回报，可承受大幅波动', score: 5 },
    ],
  },
  {
    id: 'loss',
    question: '如果投资亏损20%，您会怎么做？',
    options: [
      { label: '立即全部卖出', score: 1 },
      { label: '部分卖出，降低风险', score: 2 },
      { label: '观望等待，不做操作', score: 3 },
      { label: '逢低加仓，摊薄成本', score: 4 },
      { label: '大幅加仓，抄底机会', score: 5 },
    ],
  },
  {
    id: 'horizon',
    question: '您计划的投资期限是？',
    options: [
      { label: '1年以内', score: 1 },
      { label: '1-3年', score: 2 },
      { label: '3-5年', score: 3 },
      { label: '5-10年', score: 4 },
      { label: '10年以上', score: 5 },
    ],
  },
  {
    id: 'liquidity',
    question: '您对资金流动性的需求？',
    options: [
      { label: '随时可能需要使用', score: 1 },
      { label: '部分资金可锁定1年', score: 3 },
      { label: '大部分资金可锁定3年以上', score: 5 },
    ],
  },
]

function getRiskLevel(score: number) {
  if (score <= 12) return { level: '保守型', color: 'text-blue-500', bg: 'bg-blue-50', desc: '适合低风险、收益稳定的投资产品' }
  if (score <= 18) return { level: '稳健型', color: 'text-green-500', bg: 'bg-green-50', desc: '适合风险适中的平衡配置策略' }
  if (score <= 24) return { level: '平衡型', color: 'text-yellow-600', bg: 'bg-yellow-50', desc: '可承受一定波动，追求成长与收益平衡' }
  if (score <= 30) return { level: '成长型', color: 'text-orange-500', bg: 'bg-orange-50', desc: '追求较高回报，可承受较大波动' }
  return { level: '进取型', color: 'text-red-500', bg: 'bg-red-50', desc: '追求最大回报，可承受大幅波动' }
}

interface Props {
  onComplete: (result: { level: string; score: number }) => void
  onClose: () => void
}

export default function RiskQuestionnaire({ onComplete, onClose }: Props) {
  const [step, setStep] = useState(0)
  const [answers, setAnswers] = useState<Record<string, number>>({})
  const [showResult, setShowResult] = useState(false)

  const currentQ = QUESTIONS[step]
  const isLast = step === QUESTIONS.length - 1
  const answered = answers[currentQ.id] !== undefined

  const handleSelect = (score: number) => {
    const newAnswers = { ...answers, [currentQ.id]: score }
    setAnswers(newAnswers)
  }

  const handleNext = () => {
    if (isLast) {
      const totalScore = Object.values(answers).reduce((a, b) => a + b, 0)
      const result = getRiskLevel(totalScore)
      setShowResult(true)
      onComplete({ level: result.level, score: totalScore })
    } else {
      setStep(step + 1)
    }
  }

  const handlePrev = () => {
    if (step > 0) setStep(step - 1)
  }

  const totalScore = Object.values(answers).reduce((a, b) => a + b, 0)
  const result = getRiskLevel(totalScore)

  if (showResult) {
    return (
      <div className="message-animate flex justify-center">
        <div className="glass-card max-w-sm w-full rounded-2xl px-5 py-6 text-center">
          <div className={`mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full ${result.bg}`}>
            <ShieldCheck className={`h-7 w-7 ${result.color}`} />
          </div>
          <h3 className="mb-1 text-lg font-bold text-text-primary">风险评估结果</h3>
          <p className={`mb-2 text-2xl font-bold ${result.color}`}>{result.level}</p>
          <p className="mb-4 text-sm text-text-secondary">{result.desc}</p>
          <p className="mb-4 text-xs text-text-muted">综合得分：{totalScore} / {QUESTIONS.length * 5}</p>
          <div className="flex gap-2">
            <button onClick={() => { setShowResult(false); setStep(0); setAnswers({}) }}
              className="flex-1 rounded-lg border border-border px-3 py-2 text-xs text-text-secondary hover:bg-bg-tertiary transition-colors">
              重新评测
            </button>
            <button onClick={onClose}
              className="flex-1 rounded-lg bg-accent px-3 py-2 text-xs text-white hover:bg-accent-dark transition-colors">
              开始咨询
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="message-animate flex justify-center">
      <div className="glass-card max-w-sm w-full rounded-2xl px-5 py-5">
        {/* 进度条 */}
        <div className="mb-4 flex items-center gap-1.5">
          {QUESTIONS.map((_, i) => (
            <div
              key={i}
              className={`h-1 flex-1 rounded-full transition-colors duration-300 ${
                i < step ? 'bg-accent' : i === step ? 'bg-accent/50' : 'bg-border'
              }`}
            />
          ))}
          <span className="text-[10px] text-text-muted ml-1">{step + 1}/{QUESTIONS.length}</span>
        </div>

        <h3 className="mb-4 text-sm font-medium text-text-primary">{currentQ.question}</h3>

        <div className="space-y-2 mb-5">
          {currentQ.options.map((opt) => (
            <button
              key={opt.label}
              onClick={() => handleSelect(opt.score)}
              className={`w-full rounded-lg border px-3 py-2.5 text-left text-xs transition-all duration-200 ${
                answers[currentQ.id] === opt.score
                  ? 'border-accent bg-accent/10 text-accent font-medium'
                  : 'border-border text-text-secondary hover:border-accent/30 hover:bg-accent/5'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <div className="flex items-center justify-between">
          <button onClick={handlePrev} disabled={step === 0}
            className={`flex items-center gap-1 rounded-lg px-3 py-2 text-xs transition-colors ${
              step === 0 ? 'text-text-muted cursor-not-allowed' : 'text-text-secondary hover:text-text-primary'
            }`}>
            <ChevronLeft className="h-3.5 w-3.5" /> 上一题
          </button>
          <button onClick={handleNext} disabled={!answered}
            className={`flex items-center gap-1 rounded-lg px-4 py-2 text-xs font-medium transition-all duration-200 ${
              answered
                ? 'bg-accent text-white hover:bg-accent-dark shadow-sm'
                : 'bg-border/50 text-text-muted cursor-not-allowed'
            }`}>
            {isLast ? '查看结果' : '下一题'} <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  )
}