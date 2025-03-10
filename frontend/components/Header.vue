<template>
    <header class="background relative flex w-full">
        <div class="w-full px-6 pt-4 pb-4 flex justify-between">
            <div class="flex items-center justify-between">
                <button class="flex-none text-xl font-semibold" onclick="location.reload()">
                    Chatbot (ver2.0)
                </button>
            </div>

            <div class="dropdown flex-row items-center gap-5">
                <span
                    class="dropbtn flex items-center w-full text-gray-600 hover:text-gray-400 focus:outline-none focus:text-gray-400 font-mediums">
                    <svg width="50px" height="50px" viewBox="-1.92 -1.92 27.84 27.84" fill="none"
                        xmlns="http://www.w3.org/2000/svg" stroke="">
                        <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                        <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round" stroke="#CCCCCC"
                            stroke-width="1.7280000000000002"></g>
                        <g id="SVGRepo_iconCarrier">
                            <rect width="24" height="24" fill="#f9f9f9"></rect>
                            <circle cx="12" cy="7" r="0.5" transform="rotate(90 12 7)" stroke="#949494"
                                stroke-linecap="round" stroke-linejoin="round"></circle>
                            <circle cx="12" cy="12" r="0.5" transform="rotate(90 12 12)" stroke="#949494"
                                stroke-linecap="round" stroke-linejoin="round"></circle>
                            <circle cx="12" cy="17" r="0.5" transform="rotate(90 12 17)" stroke="#949494"
                                stroke-linecap="round" stroke-linejoin="round"></circle>
                        </g>
                    </svg>
                </span>
                <div class="dropdown-content">
                    <button @click="handleLogout">Logout</button>
                </div>
            </div>
        </div>
    </header>
</template>

<style>
.background {
    background-color: #f9f9f9;
}

.dropdown {
    position: relative;
    display: inline-block;
}

.dropdown-content {
    display: none;
    position: absolute;
    z-index: 1;
}

.dropdown-content a {
    display: block;
}

.dropdown:hover .dropdown-content {
    display: block;
}
</style>

<script setup>
import { useRouter } from '#app'

const router = useRouter()

const handleLogout = () => {
    // localStorage에서 userId 삭제
    localStorage.removeItem('userId')

    // 로그인 페이지로 리디렉션
    router.push('/login')
}
</script>
