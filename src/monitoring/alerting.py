import logging
import json
import smtplib
import requests
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """알림 심각도 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AlertConfig:
    """알림 설정"""
    # 이메일 설정
    enable_email: bool = False
    email_sender: str = ""
    email_recipients: List[str] = field(default_factory=list)
    email_server: str = "smtp.gmail.com"
    email_port: int = 587
    email_username: str = ""
    email_password: str = ""
    
    # Slack 설정
    enable_slack: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    # 웹훅 설정
    enable_webhook: bool = False
    webhook_url: str = ""
    
    # 알림 설정
    min_alert_level: AlertLevel = AlertLevel.WARNING
    throttle_interval_seconds: int = 300  # 5분
    max_alerts_per_hour: int = 10
    
    # 로깅 설정
    log_alerts: bool = True
    log_dir: Optional[str] = None

@dataclass
class Alert:
    """알림 객체"""
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    id: Optional[str] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.id is None:
            # 고유 ID 생성
            self.id = f"{self.source}_{int(self.timestamp.timestamp())}_{hash(self.title) % 10000}"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = asdict(self)
        result['level'] = self.level.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), indent=indent)


class AlertManager:
    """
    알림 관리 클래스.
    모니터링 이벤트에 대한 알림을 생성하고 전송합니다.
    """
    
    def __init__(self, config: AlertConfig):
        """
        AlertManager 초기화
        
        Args:
            config: 알림 설정
        """
        self.config = config
        
        # 알림 이력
        self.alert_history = []
        
        # 알림 제한 상태
        self.last_alert_times = {}  # alert_id -> timestamp
        self.alerts_this_hour = 0
        self.hour_start_time = datetime.now()
        
        # 커스텀 알림 처리기
        self.custom_handlers = []
        
        # 로그 디렉토리 생성
        if self.config.log_alerts and self.config.log_dir:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("AlertManager 초기화 완료")
    
    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        커스텀 알림 처리기 추가
        
        Args:
            handler: 알림을 처리할 콜백 함수 (Alert 객체 인자)
        """
        self.custom_handlers.append(handler)
    
    def alert(self, 
             level: AlertLevel, 
             title: str, 
             message: str, 
             source: str,
             details: Optional[Dict[str, Any]] = None,
             tags: Optional[List[str]] = None) -> Optional[Alert]:
        """
        알림 생성 및 전송
        
        Args:
            level: 알림 심각도 레벨
            title: 알림 제목
            message: 알림 메시지
            source: 알림 소스 (컴포넌트 또는 시스템 이름)
            details: 추가 세부 정보
            tags: 알림 태그
            
        Returns:
            생성된 Alert 객체 또는 None (알림 제한에 걸린 경우)
        """
        # 최소 레벨 확인
        if level.value < self.config.min_alert_level.value:
            return None
        
        # 알림 생성
        alert = Alert(
            level=level,
            title=title,
            message=message,
            source=source,
            details=details or {},
            tags=tags or []
        )
        
        # 알림 제한 확인
        if not self._check_alert_limits(alert):
            logger.warning(f"알림 제한에 의해 제한됨: {alert.title}")
            return None
        
        # 알림 전송
        self._send_alert(alert)
        
        # 알림 이력에 추가
        self.alert_history.append(alert)
        
        # 알림 로깅
        if self.config.log_alerts:
            self._log_alert(alert)
        
        return alert
    
    def _check_alert_limits(self, alert: Alert) -> bool:
        """
        알림 제한 확인
        
        Args:
            alert: 확인할 Alert 객체
            
        Returns:
            알림을 전송해도 되는지 여부
        """
        now = datetime.now()
        
        # 시간당 최대 알림 수 확인
        hour_diff = (now - self.hour_start_time).total_seconds() / 3600
        if hour_diff >= 1:
            # 새 시간 구간 시작
            self.hour_start_time = now
            self.alerts_this_hour = 0
        elif self.alerts_this_hour >= self.config.max_alerts_per_hour:
            # 제한 초과
            return False
        
        # 동일 알림 제한 (임계 간격 내)
        if alert.id in self.last_alert_times:
            last_time = self.last_alert_times[alert.id]
            time_diff = (now - last_time).total_seconds()
            
            if time_diff < self.config.throttle_interval_seconds:
                # 제한 간격 내
                return False
        
        # 알림 카운트 및 시간 업데이트
        self.last_alert_times[alert.id] = now
        self.alerts_this_hour += 1
        
        return True
    
    def _send_alert(self, alert: Alert) -> None:
        """
        알림 전송
        
        Args:
            alert: 전송할 Alert 객체
        """
        # 이메일 전송
        if self.config.enable_email:
            self._send_email_alert(alert)
        
        # Slack 전송
        if self.config.enable_slack:
            self._send_slack_alert(alert)
        
        # 웹훅 전송
        if self.config.enable_webhook:
            self._send_webhook_alert(alert)
        
        # 커스텀 핸들러 호출
        for handler in self.custom_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"커스텀 알림 핸들러 오류: {e}")
    
    def _send_email_alert(self, alert: Alert) -> None:
        """
        이메일로 알림 전송
        
        Args:
            alert: 전송할 Alert 객체
        """
        if not self.config.email_recipients:
            logger.warning("이메일 수신자가 설정되지 않았습니다.")
            return
        
        try:
            # 이메일 메시지 생성
            msg = MIMEMultipart()
            msg['From'] = self.config.email_sender
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            # HTML 형식 본문
            body = f"""
            <html>
              <body>
                <h2>{alert.title}</h2>
                <p><strong>심각도:</strong> {alert.level.value.upper()}</p>
                <p><strong>소스:</strong> {alert.source}</p>
                <p><strong>시간:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>메시지:</strong></p>
                <p>{alert.message}</p>
                
                <h3>세부 정보:</h3>
                <pre>{json.dumps(alert.details, indent=2)}</pre>
                
                <p><strong>태그:</strong> {', '.join(alert.tags)}</p>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # 이메일 서버 연결 및 전송
            server = smtplib.SMTP(self.config.email_server, self.config.email_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"이메일 알림 전송 완료: {alert.title}")
            
        except Exception as e:
            logger.error(f"이메일 알림 전송 오류: {e}")
    
    def _send_slack_alert(self, alert: Alert) -> None:
        """
        Slack으로 알림 전송
        
        Args:
            alert: 전송할 Alert 객체
        """
        if not self.config.slack_webhook_url:
            logger.warning("Slack 웹훅 URL이 설정되지 않았습니다.")
            return
        
        try:
            # 알림 레벨에 따른 색상
            color_map = {
                AlertLevel.INFO: "#36a64f",  # 녹색
                AlertLevel.WARNING: "#daa038",  # 주황색
                AlertLevel.ERROR: "#d00000",  # 빨간색
                AlertLevel.CRITICAL: "#7d0000"  # 진한 빨간색
            }
            
            color = color_map.get(alert.level, "#000000")
            
            # Slack 메시지 구성
            payload = {
                "channel": self.config.slack_channel,
                "attachments": [
                    {
                        "fallback": f"[{alert.level.value.upper()}] {alert.title}",
                        "color": color,
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "심각도",
                                "value": alert.level.value.upper(),
                                "short": True
                            },
                            {
                                "title": "소스",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "시간",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            },
                            {
                                "title": "태그",
                                "value": ', '.join(alert.tags) if alert.tags else "없음",
                                "short": True
                            }
                        ],
                        "footer": f"알림 ID: {alert.id}"
                    }
                ]
            }
            
            # 세부 정보가 있는 경우 첨부
            if alert.details:
                details_str = json.dumps(alert.details, indent=2)
                payload["attachments"][0]["fields"].append({
                    "title": "세부 정보",
                    "value": f"```{details_str}```",
                    "short": False
                })
            
            # 전송
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Slack 알림 전송 실패: HTTP {response.status_code} - {response.text}")
            else:
                logger.info(f"Slack 알림 전송 완료: {alert.title}")
            
        except Exception as e:
            logger.error(f"Slack 알림 전송 오류: {e}")
    
    def _send_webhook_alert(self, alert: Alert) -> None:
        """
        웹훅으로 알림 전송
        
        Args:
            alert: 전송할 Alert 객체
        """
        if not self.config.webhook_url:
            logger.warning("웹훅 URL이 설정되지 않았습니다.")
            return
        
        try:
            # 페이로드 구성
            payload = alert.to_dict()
            
            # 전송
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code < 200 or response.status_code >= 300:
                logger.warning(f"웹훅 알림 전송 실패: HTTP {response.status_code} - {response.text}")
            else:
                logger.info(f"웹훅 알림 전송 완료: {alert.title}")
            
        except Exception as e:
            logger.error(f"웹훅 알림 전송 오류: {e}")
    
    def _log_alert(self, alert: Alert) -> None:
        """
        알림 로깅
        
        Args:
            alert: 로깅할 Alert 객체
        """
        # 로그 메시지 출력
        log_message = (
            f"[{alert.level.value.upper()}] {alert.title} - {alert.message} "
            f"(소스: {alert.source}, 시간: {alert.timestamp.isoformat()})"
        )
        
        if alert.level == AlertLevel.INFO:
            logger.info(log_message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_message)
        elif alert.level in (AlertLevel.ERROR, AlertLevel.CRITICAL):
            logger.error(log_message)
        
        # 파일에 로깅 (지정된 경우)
        if self.config.log_dir:
            try:
                log_dir = Path(self.config.log_dir)
                
                # 날짜별 로그 파일
                date_str = alert.timestamp.strftime('%Y%m%d')
                log_file = log_dir / f"alerts_{date_str}.json"
                
                # JSON 형식으로 저장
                if not log_file.exists():
                    # 새 파일 생성
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump([alert.to_dict()], f, indent=2)
                else:
                    # 기존 파일에 추가
                    with open(log_file, 'r+', encoding='utf-8') as f:
                        try:
                            alerts = json.load(f)
                            alerts.append(alert.to_dict())
                            
                            # 파일 처음으로 돌아가서 덮어쓰기
                            f.seek(0)
                            f.truncate()
                            json.dump(alerts, f, indent=2)
                        except json.JSONDecodeError:
                            # 파일이 손상된 경우 새로 작성
                            f.seek(0)
                            f.truncate()
                            json.dump([alert.to_dict()], f, indent=2)
            
            except Exception as e:
                logger.error(f"알림 로깅 오류: {e}")
    
    def get_alert_history(self, 
                        level: Optional[AlertLevel] = None, 
                        source: Optional[str] = None,
                        tag: Optional[str] = None,
                        limit: int = 100) -> List[Alert]:
        """
        알림 이력 조회
        
        Args:
            level: 필터링할 알림 레벨 (None이면 모든 레벨)
            source: 필터링할 소스 (None이면 모든 소스)
            tag: 필터링할 태그 (None이면 모든 태그)
            limit: 최대 결과 수
            
        Returns:
            필터링된 알림 이력 (최신순)
        """
        # 필터링
        filtered_alerts = self.alert_history
        
        if level is not None:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        
        if source is not None:
            filtered_alerts = [a for a in filtered_alerts if a.source == source]
        
        if tag is not None:
            filtered_alerts = [a for a in filtered_alerts if tag in a.tags]
        
        # 최신순 정렬 및 제한
        return sorted(filtered_alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
    
    def clear_alert_history(self) -> None:
        """알림 이력 초기화"""
        self.alert_history = []
        self.last_alert_times = {}
        self.alerts_this_hour = 0
        self.hour_start_time = datetime.now()
