from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def generate_decision(
    analysis_state: Dict[str, Any],
    min_technical_score: int = 50,
    min_risk_reward: float = 1.5,
    use_gemini: bool = False,
    gemini_client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Motor de decisión principal. Evalúa las "Hard Rules", delega opcionalmente
    a Gemini y retorna una decisión final determinística de Trading.
    """
    
    technical = analysis_state.get('technical', {})
    score = analysis_state.get('technical_score', 0)
    risk = analysis_state.get('risk_management', {})
    events = analysis_state.get('events', {})
    
    # --- HARD RULES ---
    if not risk.get('valid', False):
        return _hold("Invalid risk parameters.")
        
    if events.get('earnings_risk_flag', False):
        return _hold(f"Earnings risk flag triggered (<= 5 days to earnings).")
        
    if technical.get('Setup', 'NONE') == 'NONE':
        return _hold("No valid technical setup detected.")
        
    if score < min_technical_score:
        return _hold(f"Technical score ({score}) below threshold ({min_technical_score}).")
        
    if risk.get('risk_reward', 0) < min_risk_reward:
        return _hold(f"Risk/Reward ({risk.get('risk_reward')}) below threshold ({min_risk_reward}).")
        
    # --- DEFAULT TECHNICAL DECISION ---
    technical_decision = 'LONG' if technical.get('Score_Is_Long', True) else 'SHORT'
    
    # --- GEMINI FILTERING (Opcional) ---
    gemini_result = None
    final_decision = technical_decision
    confidence = 1.0 # Technical determinístico
    reasoning = "All hard rules passed. Valid technical setup."
    
    if use_gemini and gemini_client:
        logger.info("Solicitando evaluación de Gemini...")
        gemini_result = gemini_client.evaluate_signal(analysis_state)
        
        if not gemini_result.get('error', False):
            # Usar Gemini como filtro restrictivo (puede degradar LONG a HOLD si ve riesgos,
            # pero no debe saltarse reglas ni inventar un LONG de la nada).
            if gemini_result['decision'] == 'HOLD':
                final_decision = 'HOLD'
                reasoning = f"Gemini vetoed the trade: {gemini_result.get('reasoning')}"
            else:
                # Si Gemini concuerda con technical_decision, lo aceptamos
                if gemini_result['decision'] == technical_decision:
                    confidence = gemini_result.get('confidence', 0.5)
                    reasoning = f"Gemini confirmed setup: {gemini_result.get('reasoning')}"
                else:
                    # Si Gemini dice SHORT pero Technical es LONG, lo vetamos (HOLD)
                    final_decision = 'HOLD'
                    reasoning = "Conflict between deterministic setup and Gemini decision."
                    
            # Si Gemini detectó divergencia fuerte, bajamos confianza
            if gemini_result.get('divergence_detected', False):
                confidence *= 0.8
                
    return {
        "decision": final_decision,
        "confidence": confidence,
        "reasoning": reasoning,
        "trade_proposal": risk if final_decision != 'HOLD' else None,
        "gemini_evaluation": gemini_result
    }

def _hold(reason: str) -> Dict[str, Any]:
    return {
        "decision": "HOLD",
        "confidence": 0.0,
        "reasoning": reason,
        "trade_proposal": None,
        "gemini_evaluation": None
    }
