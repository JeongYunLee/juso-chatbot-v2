<template>
    <div class="app-container">
        <Header />
        <div class="chat-container">

            <div class="messages-container" ref="messagesContainer">
                <ChatBubble v-for="(item, index) in messagesAndAnswers" :key="index" :message="item.message"
                    :answer="item.answer" :run_id="item.run_id" />
            </div>

            <div class="input-container">
                <form @submit.prevent="submitForm" class="input-form">
                    <input type="text" v-model="message" placeholder="질문을 입력하세요" required class="input-field"
                        :disabled="isLoading" />
                    <div class="button-container">
                        <button type="submit" class="submit-button" :disabled="isLoading">
                            {{ isLoading ? 'Sending...' : 'Send' }}
                        </button>
                        <button type="button" @click="resetSession" class="reset-button" :disabled="isLoading">
                            Reset
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue';
import ChatBubble from '../components/ChatBubble.vue';

// 기존 상태
const messagesAndAnswers = ref([]);
const message = ref('');
const isLoading = ref(false);
const messagesContainer = ref(null);

// 🆕 세션 관리 상태 (메모리만 사용 - 새로고침 시 자동 초기화)
const sessionId = ref(null);
const totalMessages = ref(0);
const showSessionInfo = ref(true); // 세션 정보 표시 여부 (개발시에만 true)

const config = useRuntimeConfig()
const baseURL = config.public.apiUrl

// 🔧 세션 관리 함수들 수정 (localStorage/sessionStorage 사용하지 않음)
const clearSession = () => {
    sessionId.value = null;
    messagesAndAnswers.value = [];
    message.value = '';
    totalMessages.value = 0;
    // console.log('🗑️ 세션 초기화');
};

// 스크롤을 맨 아래로 이동시키는 함수
const scrollToBottom = () => {
    if (messagesContainer.value) {
        const lastMessage = messagesContainer.value.lastElementChild;
        if (lastMessage) {
            lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }
};

// messagesAndAnswers가 변경될 때마다 스크롤을 맨 아래로 이동
watch(messagesAndAnswers, () => {
    nextTick(() => {
        scrollToBottom();
    });
}, { deep: true });

// 🔧 수정된 submitForm 함수
const submitForm = () => {
    if (isLoading.value) return;

    isLoading.value = true;

    // 메시지를 즉시 추가
    messagesAndAnswers.value.push({
        message: message.value,
        answer: null
    });

    const currentMessage = message.value;
    message.value = '';

    // 🔑 핵심: session_id를 요청에 포함 (null이면 서버에서 새로 생성)
    const requestBody = {
        message: currentMessage,
        session_id: sessionId.value  // null일 수 있음
    };

    // console.log('📤 메시지 전송:', {
    //     message: currentMessage.substring(0, 30) + '...',
    //     has_session_id: !!sessionId.value,
    //     session_id: sessionId.value ? sessionId.value.substring(0, 8) + '...' : 'none'
    // });

    fetch(`${import.meta.env.VITE_API_URL}/`, {
        method: "POST",
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
    })
    .then(response => response.json())
    .then(data => {
        // 🔑 핵심: 서버에서 받은 session_id 저장 (메모리에만)
        if (data.session_id) {
            sessionId.value = data.session_id;
        }
        
        // 메시지 카운트 업데이트
        if (data.message_count) {
            totalMessages.value = data.message_count;
        }

        // 응답 데이터로 메시지의 답변 업데이트
        const index = messagesAndAnswers.value.findIndex(item => item.message === currentMessage);
        if (index !== -1) {
            messagesAndAnswers.value[index].answer = data.answer;
            messagesAndAnswers.value[index].run_id = data.run_id;
        }

        // console.log('📥 응답 받음:', {
        //     session_id: sessionId.value ? sessionId.value.substring(0, 8) + '...' : 'none',
        //     message_count: data.message_count,
        //     answer_length: data.answer.length
        // });
    })
    .catch(error => {
        console.error("❌ 메시지 전송 실패:", error);
        
        // 에러 시 해당 메시지 제거 또는 에러 표시
        const index = messagesAndAnswers.value.findIndex(item => item.message === currentMessage);
        if (index !== -1) {
            messagesAndAnswers.value[index].answer = "죄송합니다. 메시지 전송 중 오류가 발생했습니다.";
        }
    })
    .finally(() => {
        isLoading.value = false;
    });
};

// 🔧 수정된 resetSession 함수
const resetSession = () => {
    if (isLoading.value) return;

    // console.log(`🔄 채팅 리셋 요청: ${sessionId.value ? sessionId.value.substring(0, 8) + '...' : 'none'}`);

    // 🔑 핵심: 현재 session_id를 리셋 요청에 포함
    const requestBody = {
        session_id: sessionId.value
    };

    fetch(`${import.meta.env.VITE_API_URL}/reset`, {
        method: "POST",
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
    })
    .then(response => response.json())
    .then(data => {
        // 🔑 핵심: 새로운 session_id로 업데이트
        if (data.session_id) {
            sessionId.value = data.session_id;
        }
        
        // UI 초기화
        messagesAndAnswers.value = [];
        message.value = '';
        totalMessages.value = 0;

        // console.log('✅ 채팅 리셋 완료:', {
        //     new_session: sessionId.value ? sessionId.value.substring(0, 8) + '...' : 'none',
        //     cleared_messages: data.cleared_messages || 0
        // });
    })
    .catch(error => {
        console.error("❌ 리셋 실패:", error);
        // 에러 발생시 로컬에서라도 초기화
        clearSession();
    });
};

// 컴포넌트 마운트 시 실행
onMounted(() => {
    // 🔧 새로고침 시 항상 새로운 세션으로 시작
    // console.log('🆕 새로운 세션으로 시작 (새로고침 감지)');
    clearSession();
});

// 컴포넌트 언마운트 시 정리 (필요시)
onUnmounted(() => {
    // 필요한 정리 작업이 있다면 여기에 추가
});
</script>

<style>
@import '../assets/css/main.css';

/* 세션 정보 스타일 (선택적) */
.session-info {
    padding: 8px 16px;
    background-color: #f5f5f5;
    border-bottom: 1px solid #ddd;
    font-family: monospace;
    color: #666;
    text-align: center;
}

.button-container {
    display: flex;
    gap: 8px;
}

.submit-button, .reset-button {
    padding: 10px 20px;
    border: none;
    border-radius: 20px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.submit-button {
    background-color: #3d98ff;
    color: white;
}

.submit-button:hover {
    background-color: #5CA3FF;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.reset-button {
    background-color: #ff3e3e;
    color: white;
}

.reset-button:hover {
    background-color: #ff5858;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.submit-button:disabled, 
.reset-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}
</style>