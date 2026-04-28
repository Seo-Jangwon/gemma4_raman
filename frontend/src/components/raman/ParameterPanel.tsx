export interface SpectrumParams {
  acqMode: 'single' | 'accumulate' | 'kinetic'
  exposureTime: number
  numAccumulations: number
  accCycleTime: number       // 0 = SDK 자동
  kineticCount: number
  kineticCycleTime: number   // 0 = SDK 자동
  readMode: 'fvb' | 'single_track' | 'image'
  preampGainIndex: number
  shutter: 'auto' | 'open' | 'close'
  laserPower: number
  targetTemp: number
  cosmicrayFilter: boolean
  accumType: 'sum' | 'avg'
  rayleighCorr: number
}

interface ParameterPanelProps {
  params: SpectrumParams
  onParamChange: (update: Partial<SpectrumParams>) => void
  availablePreampGains: number[]
  isAcquiring: boolean
  onAcquire: () => void
  onHoming: () => void
  onSilicon: () => void
}

const LBL = 'text-xs text-gray-500 mb-0.5 leading-tight'
const SEL = 'w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 bg-white'
const INP = 'w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 bg-white'
const RO  = 'w-full px-2 py-1 text-xs border border-gray-200 rounded bg-gray-100 text-gray-500 cursor-not-allowed'

function disabledInp(active: boolean) {
  return active ? INP : INP + ' opacity-40 cursor-not-allowed'
}

export default function ParameterPanel({
  params,
  onParamChange,
  availablePreampGains,
  isAcquiring,
  onAcquire,
  onHoming,
  onSilicon,
}: ParameterPanelProps) {
  const isAccum   = params.acqMode === 'accumulate'
  const isKinetic = params.acqMode === 'kinetic'
  const accumActive   = isAccum || isKinetic
  const kineticActive = isKinetic

  const preampOptions =
    availablePreampGains.length > 0
      ? availablePreampGains.map((g, i) => ({ value: i, label: `×${g}` }))
      : [
          { value: 0, label: '×1' },
          { value: 1, label: '×2' },
          { value: 2, label: '×4' },
        ]

  return (
    <div className="bg-white rounded-lg border border-gray-300 p-3 text-xs">
      {/* ── 3열 그리드 ── */}
      <div className="grid grid-cols-3 gap-x-3 gap-y-2">

        {/* ── 1열: 모드/속도/게인/셔터 ── */}
        <div className="space-y-2">
          <div>
            <div className={LBL}>Acquisition Mode</div>
            <select
              className={SEL}
              value={params.acqMode}
              onChange={e => onParamChange({ acqMode: e.target.value as SpectrumParams['acqMode'] })}
            >
              <option value="single">Single</option>
              <option value="accumulate">Accumulate</option>
              <option value="kinetic">Kinetic</option>
            </select>
          </div>

          <div>
            <div className={LBL}>Shift Speed</div>
            <input className={RO} value="16.25 μs/px" readOnly />
          </div>

          <div>
            <div className={LBL}>Readout Mode</div>
            <select
              className={SEL}
              value={params.readMode}
              onChange={e => onParamChange({ readMode: e.target.value as SpectrumParams['readMode'] })}
            >
              <option value="fvb">FVB</option>
              <option value="single_track">Single Track</option>
              <option value="image">Image</option>
            </select>
          </div>

          <div>
            <div className={LBL}>Readout Rate</div>
            <input className={RO} value="100 kHz" readOnly />
          </div>

          <div>
            <div className={LBL}>Pre-Amplifier Gain</div>
            <select
              className={SEL}
              value={params.preampGainIndex}
              onChange={e => onParamChange({ preampGainIndex: Number(e.target.value) })}
            >
              {preampOptions.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          <div>
            <div className={LBL}>Shutter</div>
            <select
              className={SEL}
              value={params.shutter}
              onChange={e => onParamChange({ shutter: e.target.value as SpectrumParams['shutter'] })}
            >
              <option value="auto">Auto</option>
              <option value="open">Open</option>
              <option value="close">Close</option>
            </select>
          </div>
        </div>

        {/* ── 2열: 시간/카운트/온도 ── */}
        <div className="space-y-2">
          <div>
            <div className={LBL}>Exposure Time (s)</div>
            <input
              type="number" className={INP} step="0.1" min="0.001"
              value={params.exposureTime}
              onChange={e => onParamChange({ exposureTime: Number(e.target.value) })}
            />
          </div>

          <div>
            <div className={LBL}>Accumulation Count</div>
            <input
              type="number" className={disabledInp(accumActive)} step="1" min="1"
              value={params.numAccumulations}
              disabled={!accumActive}
              onChange={e => onParamChange({ numAccumulations: Number(e.target.value) })}
            />
          </div>

          <div>
            <div className={LBL}>Accumulation Cycle Time (s)</div>
            <input
              type="number" className={disabledInp(accumActive)} step="0.1" min="0"
              value={params.accCycleTime}
              disabled={!accumActive}
              placeholder="0 = auto"
              onChange={e => onParamChange({ accCycleTime: Number(e.target.value) })}
            />
          </div>

          <div>
            <div className={LBL}>Kinetic Count</div>
            <input
              type="number" className={disabledInp(kineticActive)} step="1" min="1"
              value={params.kineticCount}
              disabled={!kineticActive}
              onChange={e => onParamChange({ kineticCount: Number(e.target.value) })}
            />
          </div>

          <div>
            <div className={LBL}>Kinetic Cycle Time (s)</div>
            <input
              type="number" className={disabledInp(kineticActive)} step="0.1" min="0"
              value={params.kineticCycleTime}
              disabled={!kineticActive}
              placeholder="0 = auto"
              onChange={e => onParamChange({ kineticCycleTime: Number(e.target.value) })}
            />
          </div>

          <div>
            <div className={LBL}>Temperature (°C)</div>
            <input
              type="number" className={INP} step="1"
              value={params.targetTemp}
              onChange={e => onParamChange({ targetTemp: Number(e.target.value) })}
            />
          </div>
        </div>

        {/* ── 3열: 그레이팅/레이저/파워/버튼 ── */}
        <div className="space-y-2">
          <div>
            <div className={LBL}>Grating</div>
            <input className={RO} value="1200 gr/mm" readOnly />
          </div>

          <div>
            <div className={LBL}>Unit</div>
            <input className={RO} value="cm⁻¹" readOnly />
          </div>

          <div>
            <div className={LBL}>Laser</div>
            <input className={RO} value="532 nm" readOnly />
          </div>

          <div>
            <div className={LBL}>Power (%)</div>
            <select
              className={SEL}
              value={params.laserPower}
              onChange={e => onParamChange({ laserPower: Number(e.target.value) })}
            >
              {[20, 40, 60, 80, 100].map(p => (
                <option key={p} value={p}>{p}%</option>
              ))}
            </select>
          </div>

          {/* 파워 빠른 선택 */}
          <div className="flex gap-0.5">
            {[20, 40, 60, 80, 100].map(p => (
              <button
                key={p}
                onClick={() => onParamChange({ laserPower: p })}
                title={`${p}%`}
                className={`flex-1 py-1 text-xs rounded border transition-colors font-medium ${
                  params.laserPower === p
                    ? 'bg-blue-600 border-blue-600 text-white'
                    : 'bg-gray-700 border-gray-600 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {p}
              </button>
            ))}
          </div>

          {/* 액션 버튼 */}
          <div className="grid grid-cols-3 gap-1">
            <button
              onClick={onHoming}
              className="py-1 text-xs bg-gray-700 text-white rounded hover:bg-gray-500 transition-colors"
            >
              Homing
            </button>
            <button
              className="py-1 text-xs bg-gray-700 text-white rounded hover:bg-gray-500 transition-colors"
            >
              Calibrate
            </button>
            <button
              onClick={onSilicon}
              className="py-1 text-xs bg-gray-700 text-white rounded hover:bg-gray-500 transition-colors"
            >
              Silicon
            </button>
          </div>

          <div>
            <div className={LBL}>Rayleigh Corr. (nm)</div>
            <input
              type="number" className={INP} step="0.001"
              value={params.rayleighCorr}
              onChange={e => onParamChange({ rayleighCorr: Number(e.target.value) })}
            />
          </div>
        </div>
      </div>

      {/* ── 하단 바: 코스믹레이 + 누적 타입 + ACQUIRE ── */}
      <div className="mt-3 pt-3 border-t border-gray-200 flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-5">
          {/* Cosmicray Filter 토글 */}
          <label className="flex items-center gap-2 cursor-pointer">
            <button
              type="button"
              role="switch"
              aria-checked={params.cosmicrayFilter}
              onClick={() => onParamChange({ cosmicrayFilter: !params.cosmicrayFilter })}
              className={`relative w-9 h-5 rounded-full transition-colors focus:outline-none ${
                params.cosmicrayFilter ? 'bg-blue-500' : 'bg-gray-300'
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                  params.cosmicrayFilter ? 'translate-x-4' : 'translate-x-0'
                }`}
              />
            </button>
            <span className="text-gray-600">Cosmicray Filter</span>
          </label>

          {/* 누적 타입 */}
          <div className="flex items-center gap-3 text-gray-600">
            <span>Accumulation Type</span>
            {(['sum', 'avg'] as const).map(t => (
              <label key={t} className="flex items-center gap-1 cursor-pointer">
                <input
                  type="radio"
                  name="accumType"
                  value={t}
                  checked={params.accumType === t}
                  onChange={() => onParamChange({ accumType: t })}
                  className="w-3 h-3 accent-blue-600"
                />
                <span>{t === 'sum' ? 'Sum' : 'Avg'}</span>
              </label>
            ))}
          </div>
        </div>

        {/* ACQUIRE 버튼 */}
        <button
          onClick={onAcquire}
          disabled={isAcquiring}
          className="px-6 py-2 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2 min-w-[120px] justify-center"
        >
          {isAcquiring ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin inline-block" />
              측정 중...
            </>
          ) : (
            <>▶&nbsp;ACQUIRE</>
          )}
        </button>
      </div>
    </div>
  )
}
